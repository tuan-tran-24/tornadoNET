import os
import sys
import re
import time
import uuid
from pathlib import Path

import arcpy
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# --------------------------------------------------------
#  Hard-coded values
# --------------------------------------------------------

NUM_CLASSES = 4

OUT_ID_TO_NAME = {
    -1: "Unknown",
     0: "Uncertain",
     1: "Uninhabited",
     2: "Empty",
     3: "Rebuilding",
     4: "Rebuilt",
}

LOGITS_INDEX         = [0, 1, 2, 3]
WEIGHT_SEQUENCE      = 0.6
SEQUENCE_LEN         = 16
STRIDE               = 16
MAX_FRAMES_PER_RUN   = 128
STILL_BAG_SIZE       = 64

# --------------------------------------------------------
#  Helper functions
# --------------------------------------------------------

def normalize_text(text):
    if text is None:
        return ""
    text = re.sub(r"[^A-Z0-9]+", " ", str(text).upper())
    return re.sub(r"\s+", " ", text).strip()

def extract_pin_from_text(text):
    if not text:
        return None
    match = re.search(r"(\d{10,})\s*$", str(text))
    return match.group(1) if match else None

def extract_pin_from_folder_name(folder_name):
    if not folder_name:
        return None

    numbers = re.findall(r"\d+", str(folder_name))
    if not numbers:
        return None

    candidate = numbers[-1]
    return candidate if len(candidate) >= 10 else None

def has_real_street_number(normalized_address):
    if not normalized_address:
        return False

    tokens = normalized_address.split()
    numeric_tokens = [token for token in tokens if token.isdigit()]
    if not numeric_tokens:
        return False

    for token in numeric_tokens:
        try:
            value = int(token)
            if value >= 1 and len(token) <= 6:
                return True
        except Exception:
            pass

    return False

def list_images_recursive(address_folder):
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = []

    for root, dirs, files in os.walk(address_folder):
        dirs[:] = [directory for directory in dirs if directory.lower() != "surrounding"]

        for filename in files:
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, filename))

    return image_paths

def is_address_folder_name(folder_name):
    if not folder_name:
        return False

    normalized_name = normalize_text(folder_name)

    ignored_folder_names = {
        "EMPTY",
        "REBUILT",
        "REBUILDING",
        "UNINHABITED",
        "OTHER",
        "SURROUNDING",
    }

    if normalized_name in ignored_folder_names:
        return False

    return True

def iter_address_folders(selected_roots):
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for root in selected_roots:
        if not root or not os.path.isdir(root):
            continue

        for current_dir, dirs, files in os.walk(root):
            dirs[:] = [directory for directory in dirs if directory.lower() != "surrounding"]
            current_name = os.path.basename(current_dir)

            direct_images = [
                os.path.join(current_dir, filename)
                for filename in files
                if os.path.splitext(filename)[1].lower() in image_extensions
            ]

            is_selected_root = os.path.normpath(current_dir) == os.path.normpath(root)
            root_can_be_address = bool(direct_images) or extract_pin_from_folder_name(current_name) is not None

            if is_selected_root and not root_can_be_address:
                continue

            if is_address_folder_name(current_name):
                frame_paths = list_images_recursive(current_dir)

                if frame_paths:
                    yield current_name, current_dir, frame_paths

                dirs[:] = []

def find_field_case_insensitive(feature_class, wanted_name):
    wanted_key = re.sub(r"[\s_]+", "", wanted_name).lower()

    for field in arcpy.ListFields(feature_class) or []:
        field_key = re.sub(r"[\s_]+", "", field.name).lower()
        if field_key == wanted_key:
            return field.name

    return None


def safe_feature_class_base(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", os.path.basename(name))[:120]


def verify_exists(path, tries=20, delay=0.2):
    for _ in range(tries):
        if arcpy.Exists(path):
            return True
        time.sleep(delay)
    return arcpy.Exists(path)

def safe_feature_class_name(name, gdb):
    try:
        return arcpy.ValidateTableName(os.path.basename(name), gdb)
    except Exception:
        return re.sub(r"[^A-Za-z0-9_]+", "_", os.path.basename(name))[:120]

def clear_workspace_locks():
    try:
        arcpy.ClearWorkspaceCache_management()
    except Exception:
        pass
    time.sleep(0.15)

def force_enable_attachments(feature_class_path):
    description = arcpy.Describe(feature_class_path)
    gdb = description.path

    if not gdb or ".gdb" not in gdb.lower():
        raise arcpy.ExecuteError("Attachments require a file geodatabase output.")

    old_workspace = arcpy.env.workspace
    arcpy.env.workspace = gdb

    safe_name = safe_feature_class_name(feature_class_path, gdb)
    safe_feature_class_path = os.path.join(gdb, safe_name)

    if os.path.normpath(safe_feature_class_path) != os.path.normpath(feature_class_path):
        arcpy.management.Rename(feature_class_path, safe_name, "FeatureClass")
        feature_class_path = safe_feature_class_path

    attachment_table = os.path.join(gdb, f"{safe_name}__ATTACH")
    attachment_relationship = os.path.join(gdb, f"{safe_name}__ATTACHREL")

    for path in (attachment_relationship, attachment_table):
        if arcpy.Exists(path):
            try:
                arcpy.management.Delete(path)
            except Exception:
                pass

    if not getattr(arcpy.Describe(feature_class_path), "hasGlobalID", False):
        arcpy.management.AddGlobalIDs(feature_class_path)

    try:
        if getattr(arcpy.Describe(feature_class_path), "hasAttachments", False):
            arcpy.management.DisableAttachments(feature_class_path)
    except Exception:
        pass

    clear_workspace_locks()
    try:
        arcpy.management.EnableAttachments(feature_class_path)
    except Exception:
        clear_workspace_locks()
        for path in (attachment_relationship, attachment_table):
            if arcpy.Exists(path):
                try:
                    arcpy.management.Delete(path)
                except Exception:
                    pass
        arcpy.management.EnableAttachments(feature_class_path)

    arcpy.env.workspace = old_workspace

    if not (arcpy.Exists(attachment_table) and arcpy.Exists(attachment_relationship)):
        raise arcpy.ExecuteError("Attachment tables were not created.")

    return feature_class_path

def make_local_scratch_gdb():
    base_folder = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    run_folder = os.path.join(
        base_folder,
        "ParcelModelAttach_Temp",
        f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
    )
    os.makedirs(run_folder, exist_ok=True)

    scratch_gdb = os.path.join(run_folder, "scratch.gdb")
    arcpy.management.CreateFileGDB(run_folder, "scratch")
    return scratch_gdb

def copy_back_if_possible(temp_feature_class, desired_feature_class, retries=6, delay=1.25):
    if os.path.normpath(temp_feature_class) == os.path.normpath(desired_feature_class):
        return temp_feature_class

    target_gdb = os.path.dirname(desired_feature_class)
    target_name = safe_feature_class_name(desired_feature_class, target_gdb)
    final_target = os.path.join(target_gdb, target_name)

    for attempt in range(retries):
        try:
            if arcpy.Exists(final_target):
                arcpy.management.Delete(final_target)

            arcpy.env.maintainAttachments = True
            try:
                arcpy.management.CopyFeatures(temp_feature_class, final_target)
            except Exception:
                arcpy.conversion.FeatureClassToFeatureClass(
                    temp_feature_class,
                    target_gdb,
                    target_name,
                )

            if verify_exists(final_target):
                return final_target
        except Exception as error:
            arcpy.AddWarning(f"[CopyBack {attempt + 1}/{retries}] {error}")

        time.sleep(delay)

    return temp_feature_class

def _transform(image_size=256):
    return transforms.Compose([
        transforms.Resize((int(image_size), int(image_size))),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def frame_sort_key(path):
    basename = os.path.basename(path)

    match = re.search(r"(?:frame[_\-\s]?)(\d+)", basename, flags=re.IGNORECASE)
    if match:
        try:
            return (0, int(match.group(1)))
        except Exception:
            pass

    match = re.search(r"(\d+)", basename)
    if match:
        try:
            return (1, int(match.group(1)))
        except Exception:
            pass

    return (2, basename.lower())

def subsample_even(paths, max_count):
    if (max_count is None) or (max_count <= 0) or (len(paths) <= max_count):
        return paths

    total_count = len(paths)
    step = float(total_count) / float(max_count)

    selected_indices = []
    seen = set()

    for i in range(max_count):
        index = int(i * step)
        index = max(0, min(index, total_count - 1))
        if index not in seen:
            selected_indices.append(index)
            seen.add(index)

    index = total_count - 1
    while len(selected_indices) < max_count and index >= 0:
        if index not in seen:
            selected_indices.append(index)
            seen.add(index)
        index -= 1

    selected_indices.sort()
    return [paths[index] for index in selected_indices]

# --------------------------------------------------------
#  Set up the model
# --------------------------------------------------------
def strip_state_dict_prefixes(state_dict):
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]
        if new_key.startswith("net."):
            new_key = new_key[4:]
        if new_key.startswith("model."):
            new_key = new_key[6:]

        cleaned_state_dict[new_key] = value

    return cleaned_state_dict

class cnn_lstm(nn.Module):
    def __init__(self,
                 num_classes: int = NUM_CLASSES,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()

        # ResNet50 backbone
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()  # (N,2048)
        self.backbone = backbone

        # LSTM Module
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )

        # fully-connected layers
        self.head_sequence = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

        # fully-connected layers
        self.head_still = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes),
        )

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:          # [T,3,H,W]
            x = x.unsqueeze(0)    # [1,T,3,H,W]

        N, T, C, H, W = x.shape

        # performs convolutional operation on each frame
        features = self.backbone(
            x.reshape(N * T, C, H, W)
        ).reshape(N, T, 2048)  # [N,T,2048]

        # sequential modeling
        out, _ = self.lstm(features)  # [N,T,H]

        # aggregate across time and then mean
        h_mean = out.mean(dim=1)   # [N,H]
        logits_sequence = self.head_sequence(h_mean)
        return logits_sequence        # [N,K]

    def forward_still(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5: # [1,K,3,H,W]
            x = x.squeeze(0) # [K,3,H,W]

        if x.dim() == 3: # [3,H,W]
            x = x.unsqueeze(0) # [1,3,H,W]

        # pass stills to backbone and then mean
        features = self.backbone(x)            # [K,2048]
        logits_still = self.head_still(features)     # [K,C]

        # safeguard to ensure same dimension
        if logits_still.shape[0] > 1:
            logits_still = logits_still.mean(dim=0,keepdim=True)

        return logits_still

def unwrap_logits(model_output):
    if isinstance(model_output, (tuple, list)) and len(model_output) > 0:
        return model_output[0]

    if isinstance(model_output, dict):
        for key in ("logits", "yhat", "out"):
            if key in model_output:
                return model_output[key]

    return model_output

def sequence_infer(
    model,
    device,
    transform,
    run_frame_paths,
    sequence_len,
    stride,
    max_frames_per_run,
):
    if not run_frame_paths:
        return None, 0

    run_frame_paths = sorted(run_frame_paths, key=frame_sort_key)
    run_frame_paths = subsample_even(run_frame_paths, max_frames_per_run)

    transformed_frames = []
    for frame_path in run_frame_paths:
        if not os.path.isfile(frame_path):
            continue
        try:
            image = Image.open(frame_path).convert("RGB")
            transformed_frames.append(transform(image))
        except Exception:
            continue

    num_frames = len(transformed_frames)
    if num_frames == 0:
        return None, 0

    if sequence_len <= 0:
        sequence_len = 16
    if stride <= 0:
        stride = sequence_len

    if num_frames < sequence_len:
        padded_frames = transformed_frames + [transformed_frames[-1]] * (sequence_len - num_frames)
        windows = [padded_frames]
    else:
        window_starts = list(range(0, num_frames - sequence_len + 1, stride))
        if not window_starts:
            window_starts = [0]
        windows = [
            transformed_frames[start:start + sequence_len]
            for start in window_starts
        ]

    logits_sum = None
    num_windows = 0

    with torch.no_grad():
        for window_frames in windows:
            sequence_tensor = torch.stack(window_frames, dim=0).unsqueeze(0).to(device, non_blocking=True)

            if hasattr(model, "forward_sequence"):
                model_output = model.forward_sequence(sequence_tensor)
            else:
                model_output = model(sequence_tensor)

            logits = unwrap_logits(model_output)

            if logits is None:
                continue
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            logits_sum = logits if logits_sum is None else (logits_sum + logits)
            num_windows += 1

    if logits_sum is None or num_windows == 0:
        return None, num_frames

    run_logits = logits_sum / float(num_windows)
    return run_logits, num_frames

def still_infer(
    model,
    device,
    transform,
    all_frame_paths,
    still_bag_size,
):
    if not all_frame_paths:
        return None

    all_frame_paths = sorted(all_frame_paths, key=frame_sort_key)
    all_frame_paths = subsample_even(all_frame_paths, still_bag_size)

    transformed_frames = []
    for frame_path in all_frame_paths:
        if not os.path.isfile(frame_path):
            continue
        try:
            image = Image.open(frame_path).convert("RGB")
            transformed_frames.append(transform(image))
        except Exception:
            continue

    if not transformed_frames:
        return None

    still_tensor = torch.stack(transformed_frames, dim=0).unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad():
        if hasattr(model, "forward_still"):
            model_output = model.forward_still(still_tensor)
        else:
            model_output = model(still_tensor)

        logits = unwrap_logits(model_output)
        if logits is None:
            return None
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

    return logits

def hybrid_infer(
    model,
    device,
    keep_idx,
    threshold,
    runs,
    all_frames,
    weight_sequence=0.6,
    sequence_len=16,
    stride=16,
    max_frames_per_run=128,
    still_bag_size=64,
    img_size=256,
):
    transform = _transform(image_size=img_size)

    run_logits_list = []
    run_lengths = []

    for run_info in runs or []:
        run_frame_paths = run_info.get("frames") or []

        run_logits, used_run_length = sequence_infer(
            model=model,
            device=device,
            transform=transform,
            run_frame_paths=run_frame_paths,
            sequence_len=sequence_len,
            stride=stride,
            max_frames_per_run=max_frames_per_run,
        )

        if run_logits is None:
            continue

        run_logits_list.append(run_logits.squeeze(0))
        run_lengths.append(float(used_run_length))

    have_sequence = len(run_logits_list) > 0
    sequence_logits = None

    if have_sequence:
        run_weights = torch.tensor(run_lengths, device=device, dtype=torch.float32)
        run_weights = run_weights / run_weights.sum().clamp_min(1.0)
        run_weights = run_weights.unsqueeze(1)

        stacked_logits = torch.stack(run_logits_list, dim=0)
        sequence_logits = (stacked_logits * run_weights).sum(dim=0, keepdim=True)

    still_logits = still_infer(
        model=model,
        device=device,
        transform=transform,
        all_frame_paths=all_frames,
        still_bag_size=still_bag_size,
    )
    have_still = still_logits is not None

    weight_sequence = max(0.0, min(1.0, float(weight_sequence)))
    w_still = 1.0 - weight_sequence
    if have_sequence and have_still:
        fused_logits = (weight_sequence * sequence_logits) + (w_still * still_logits)
    elif have_sequence:
        fused_logits = sequence_logits
    elif have_still:
        fused_logits = still_logits
    else:
        return -1, 0.0, False, False, 0

    try:
        logits_kept = fused_logits[:, keep_idx]
    except Exception as error:
        raise arcpy.ExecuteError(
            f"Failed to index logits with keep_idx={keep_idx}. Error: {error}"
        )

    probabilities = torch.softmax(logits_kept.squeeze(0), dim=0)
    confidence = float(probabilities.max().item())
    predicted_index = int(probabilities.argmax().item())
    predicted_id = predicted_index + 1

    if confidence < float(threshold):
        return 0, confidence, have_sequence, have_still, len(run_logits_list)

    return predicted_id, confidence, have_sequence, have_still, len(run_logits_list)

class Toolbox(object):
    def __init__(self):
        self.label = "Perform Recovery Assessment"
        self.alias = "PerformRecoveryAssessment"
        self.tools = [PerformRecoveryAssessment]

class PerformRecoveryAssessment(object):
    def __init__(self):
        self.label = "Perform Recovery Assessment"
        self.description = (
            "Perform inference per parcel, assign recovery score, and attach frames"
        )

    def getParameterInfo(self):
        address_roots = arcpy.Parameter(
            displayName="Address Folders",
            name="address_roots",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input",
        )
        address_roots.multiValue = True

        parcel_layers = arcpy.Parameter(
            displayName="Parcels",
            name="in_parcels",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        parcel_layers.filter.list = ["Polygon"]
        parcel_layers.multiValue = True

        model_file = arcpy.Parameter(
            displayName="Load Model (.pth)",
            name="model_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input",
        )
        model_file.filter.list = ["pt", "pth"]

        confidence_threshold = arcpy.Parameter(
            displayName="Confidence Threshold",
            name="conf_thresh",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
        )
        confidence_threshold.value = 0.50

        attach_frames = arcpy.Parameter(
            displayName="Attach frames to parcels",
            name="do_attachments",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        attach_frames.value = True

        output_feature_class = arcpy.Parameter(
            displayName="Output Parcels",
            name="out_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
        )

        return [
            address_roots,
            parcel_layers,
            model_file,
            confidence_threshold,
            attach_frames,
            output_feature_class,
        ]

    def updateMessages(self, params):
        model_path = params[2].valueAsText
        if model_path and not os.path.isfile(model_path):
            params[2].setErrorMessage("Pick an existing model file.")

        threshold_value = params[3].value
        if threshold_value is not None and (threshold_value < 0 or threshold_value > 1):
            params[3].setErrorMessage("Threshold must be between 0 and 1.")

        output_fc = params[5].valueAsText
        if output_fc and (not os.path.dirname(output_fc).lower().endswith(".gdb")):
            params[5].setErrorMessage("Output must be inside a File GDB (*.gdb).")

    def execute(self, params, messages):
        arcpy.env.addOutputsToMap = False
        arcpy.env.maintainAttachments = True

        selected_roots = []
        for value in (params[0].values or []):
            if value is None:
                continue
            selected_roots.append(value.value if hasattr(value, "value") else str(value))

        if not selected_roots:
            raise arcpy.ExecuteError("No address roots provided.")

        parcel_input_values = params[1].values
        if not parcel_input_values:
            raise arcpy.ExecuteError("No parcels provided.")

        parcel_layers = [value for value in parcel_input_values if value is not None]
        if not parcel_layers:
            raise arcpy.ExecuteError("No valid parcel layers found.")

        model_path = params[2].valueAsText
        confidence_threshold = float(params[3].value)
        attach_frames = bool(params[4].value)
        desired_output_feature_class = params[5].valueAsText

        keep_indices = LOGITS_INDEX
        weight_sequence = WEIGHT_SEQUENCE
        sequence_length = SEQUENCE_LEN
        stride = STRIDE
        max_frames_per_run = MAX_FRAMES_PER_RUN
        still_bag_size = STILL_BAG_SIZE

        if len(parcel_layers) == 1:
            source_parcels = parcel_layers[0]
        else:
            temp_gdb = make_local_scratch_gdb()
            source_parcels = os.path.join(temp_gdb, "MergedParcels")
            arcpy.management.Merge(parcel_layers, source_parcels)
            if not verify_exists(source_parcels):
                raise arcpy.ExecuteError("Failed to merge input parcel layers.")

        target_gdb = os.path.dirname(desired_output_feature_class)
        output_name = safe_feature_class_base(os.path.basename(desired_output_feature_class))
        output_feature_class = os.path.join(target_gdb, output_name)

        if arcpy.Exists(output_feature_class):
            arcpy.management.Delete(output_feature_class)

        created = False
        try:
            arcpy.management.CopyFeatures(source_parcels, output_feature_class)
            created = verify_exists(output_feature_class)
        except Exception:
            created = False

        if not created:
            try:
                arcpy.conversion.FeatureClassToFeatureClass(
                    source_parcels,
                    target_gdb,
                    output_name,
                )
                created = verify_exists(output_feature_class)
            except Exception:
                created = False

        if not created:
            scratch_gdb = make_local_scratch_gdb()
            output_feature_class = os.path.join(scratch_gdb, output_name)

            try:
                arcpy.management.CopyFeatures(source_parcels, output_feature_class)
            except Exception:
                arcpy.conversion.FeatureClassToFeatureClass(
                    source_parcels,
                    scratch_gdb,
                    output_name,
                )

            if not verify_exists(output_feature_class):
                raise arcpy.ExecuteError("Could not create output feature class.")

        existing_fields = {field.name.lower(): field for field in (arcpy.ListFields(output_feature_class) or [])}
        if "recovery_score" not in existing_fields:
            arcpy.management.AddField(output_feature_class, "recovery_score", "SHORT")
        if "recovery_state" not in existing_fields:
            arcpy.management.AddField(output_feature_class, "recovery_state", "TEXT", field_length=32)
        if "prediction_confidence" not in existing_fields:
            arcpy.management.AddField(output_feature_class, "prediction_confidence", "DOUBLE")

        if attach_frames and "image_path" not in existing_fields:
            arcpy.management.AddField(output_feature_class, "Image_Path", "TEXT", field_length=1000)

        oid_field = arcpy.Describe(output_feature_class).OIDFieldName
        address_field = (
            find_field_case_insensitive(output_feature_class, "Own_Addres")
            or find_field_case_insensitive(output_feature_class, "Address")
        )
        pin_field = find_field_case_insensitive(output_feature_class, "PIN")

        if not address_field:
            raise arcpy.ExecuteError(
                "Could not find an address field such as 'Own_Addres' or 'Address'."
            )

        # --------------------------------------------------------
        # Only keep important fields
        # --------------------------------------------------------
        desc = arcpy.Describe(output_feature_class)
        shape_field = desc.shapeFieldName

        keep_fields = {
            oid_field.lower(),
            shape_field.lower(),
            "recovery_score",
            "recovery_state",
            "prediction_confidence",
        }

        if pin_field:
            keep_fields.add(pin_field.lower())
        if address_field:
            keep_fields.add(address_field.lower())
        if attach_frames:
            keep_fields.add("image_path")

        drop_fields = []
        for field in arcpy.ListFields(output_feature_class):
            if field.required:
                continue
            if field.name.lower() not in keep_fields:
                drop_fields.append(field.name)

        if drop_fields:
            arcpy.management.DeleteField(output_feature_class, drop_fields)


        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = cnn_lstm(
            num_classes=NUM_CLASSES,
            lstm_hidden=128,
            lstm_layers=1,
            dropout=0.2,
        )

        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                raise arcpy.ExecuteError("Unsupported checkpoint format.")

            state_dict = strip_state_dict_prefixes(state_dict)
            model.load_state_dict(state_dict, strict=True)

        model.to(device)
        model.eval()

        pin_buckets = {}
        address_buckets = {}

        arcpy.SetProgressor("marquee", "Scanning address folders...")

        for folder_name, folder_path, frame_paths in iter_address_folders(selected_roots):
            pin_value = extract_pin_from_folder_name(folder_name)
            normalized_name = normalize_text(folder_name)
            valid_frame_paths = [frame_path for frame_path in frame_paths if os.path.isfile(frame_path)]

            if not valid_frame_paths:
                continue

            bucket = address_buckets.get(normalized_name)

            if bucket is None:
                bucket = {
                    "pin": pin_value,
                    "address_key": normalized_name,
                    "runs": [],
                    "frame_set": set(),
                    "sample_name": folder_name,
                }
                address_buckets[normalized_name] = bucket

            if pin_value and not bucket.get("pin"):
                bucket["pin"] = pin_value

            bucket["runs"].append({
                "name": folder_name,
                "frames": valid_frame_paths,
            })

            for frame_path in valid_frame_paths:
                bucket["frame_set"].add(frame_path)

            # Optional fallback key: PIN. Address matching is attempted first later.
            if pin_value:
                pin_buckets[pin_value] = bucket

        arcpy.ResetProgressor()

        unique_buckets = list(address_buckets.values())
        total_buckets = len(unique_buckets)

        arcpy.SetProgressor("step", "Performing Recovery Assessment...", 0, total_buckets, 1)

        for i, bucket in enumerate(unique_buckets, 1):
            arcpy.SetProgressorLabel(f"Performing Recovery Assessment {i}/{total_buckets}")
            arcpy.SetProgressorPosition(i)

            all_frames = sorted(bucket["frame_set"])
            runs = bucket["runs"]

            recovery_score, confidence, have_sequence, have_still, num_runs_used = hybrid_infer(
                model=model,
                device=device,
                keep_idx=keep_indices,
                threshold=confidence_threshold,
                runs=runs,
                all_frames=all_frames,
                weight_sequence=weight_sequence,
                sequence_len=sequence_length,
                stride=stride,
                max_frames_per_run=max_frames_per_run,
                still_bag_size=still_bag_size,
                img_size=256,
            )

            bucket["recovery_score"] = int(recovery_score)
            bucket["recovery_state"] = OUT_ID_TO_NAME.get(int(recovery_score), "Unknown")
            bucket["prediction_confidence"] = float(confidence)
            bucket["frame_list"] = all_frames
            bucket["have_sequence"] = bool(have_sequence)
            bucket["have_still"] = bool(have_still)
            bucket["num_runs_used"] = int(num_runs_used)

        arcpy.ResetProgressor()

        rows_for_attachments = []

        fields = [oid_field, address_field] + ([pin_field] if pin_field else []) + [
            "recovery_score", "recovery_state", "prediction_confidence",
        ]

        if attach_frames:
            fields.append("Image_Path")

        field_index = {field_name: i for i, field_name in enumerate(fields)}

        parcel_count = int(arcpy.management.GetCount(output_feature_class)[0])

        arcpy.SetProgressor("step", "Writing predictions to parcels...", 0, parcel_count, 1)

        with arcpy.da.UpdateCursor(output_feature_class, fields) as cursor:
            for i, row in enumerate(cursor,1):
                arcpy.SetProgressorLabel(f"Updating parcel {i}/{parcel_count}")
                arcpy.SetProgressorPosition(i)

                object_id = int(row[field_index[oid_field]])
                address_value = row[field_index[address_field]]
                pin_value = row[field_index[pin_field]] if pin_field else None

                normalized_address = normalize_text(address_value) if address_value else None
                normalized_pin = normalize_text(pin_value) if pin_value else None

                address_plus_pin = (
                    normalize_text(f"{address_value} {pin_value}")
                    if address_value and pin_value
                    else None
                )

                matched_bucket = address_buckets.get(normalized_address) if normalized_address else None

                if not matched_bucket and address_plus_pin:
                    matched_bucket = address_buckets.get(address_plus_pin)

                if not matched_bucket:
                    pin_key = (
                        extract_pin_from_text(pin_value)
                        or extract_pin_from_text(address_value)
                        or normalized_pin
                    )
                    matched_bucket = pin_buckets.get(pin_key) if pin_key else None

                if matched_bucket and "recovery_score" in matched_bucket:
                    row[field_index["recovery_score"]] = int(matched_bucket["recovery_score"])
                    row[field_index["recovery_state"]] = matched_bucket["recovery_state"]
                    row[field_index["prediction_confidence"]] = float(matched_bucket["prediction_confidence"])

                    if attach_frames:
                        image_paths = [
                            frame_path for frame_path in matched_bucket["frame_list"]
                            if os.path.isfile(frame_path)
                        ]
                        row[field_index["Image_Path"]] = image_paths[0] if image_paths else None

                        for frame_path in image_paths:
                            rows_for_attachments.append((object_id, frame_path))

                else:
                    row[field_index["recovery_score"]] = -1
                    row[field_index["recovery_state"]] = "Unknown"
                    row[field_index["prediction_confidence"]] = None
                    if attach_frames:
                        row[field_index["Image_Path"]] = None

                cursor.updateRow(row)

        arcpy.ResetProgressor()

        if attach_frames:
            output_feature_class = force_enable_attachments(output_feature_class)

            if rows_for_attachments:
                temp_table = arcpy.CreateUniqueName("att_tbl", "in_memory")
                arcpy.management.CreateTable("in_memory", os.path.basename(temp_table))
                arcpy.management.AddField(temp_table, "KEYOID", "LONG")
                arcpy.management.AddField(temp_table, "ATTACHMENT", "TEXT", field_length=1000)

                total_attachments = len(rows_for_attachments)

                arcpy.SetProgressor("step", "Adding attachments...", 0, total_attachments, 1)

                with arcpy.da.InsertCursor(temp_table, ["KEYOID", "ATTACHMENT"]) as cursor:
                    for i, (object_id, frame_path) in enumerate(rows_for_attachments, 1):
                        arcpy.SetProgressorLabel(f"Preparing attachment {i}/{total_attachments}")
                        arcpy.SetProgressorPosition(i)
                        cursor.insertRow((object_id, frame_path))

                arcpy.ResetProgressor()

                arcpy.management.AddAttachments(
                    output_feature_class,
                    oid_field,
                    temp_table,
                    "KEYOID",
                    "ATTACHMENT",
                )

                try:
                    arcpy.management.Delete(temp_table)
                except Exception:
                    pass

        final_output = copy_back_if_possible(
            output_feature_class,
            desired_output_feature_class,
        )
        params[5].value = final_output