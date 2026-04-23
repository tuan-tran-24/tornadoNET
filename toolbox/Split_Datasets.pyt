import arcpy
import os
import re
import shutil
import uuid
from collections import defaultdict
from difflib import SequenceMatcher


# -----------------------------------------------------
# Hard-code values
# -----------------------------------------------------
CATEGORIES = {"other", "uninhabited", "empty", "rebuilding", "rebuilt"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------
def _normalize_path(path):
    return os.path.normpath(os.path.abspath(path))

def _ensure_folder(folder_path):
    if folder_path and not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return folder_path

def _msg(text): arcpy.AddMessage(text)
def _warn(text): arcpy.AddWarning(text)
def _err(text): arcpy.AddError(text)

def _canon_key(text):
    if text is None:
        return ""
    text = str(text).upper()
    text = re.sub(r"[^A-Z0-9]+", "", text)
    return text

def _safe_name(text, max_length=150):
    if text is None:
        return ""
    text = re.sub(r"[\\/:*?\"<>|]+", "_", str(text).strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_length]

def _find_existing_field(feature_class, candidate_names):
    field_names = [f.name for f in arcpy.ListFields(feature_class)]
    lookup = {f.lower(): f for f in field_names}

    for name in candidate_names:
        key = name.lower()
        if key in lookup:
            return lookup[key]
    return None

def _unique_name(name, workspace="in_memory"):
    return arcpy.CreateUniqueName(name, workspace)

def _strip_wrapping_quotes(text):
    if text is None:
        return text
    text = str(text).strip()
    if len(text) >= 2 and ((text[0] == "'" and text[-1] == "'") or (text[0] == '"' and text[-1] == '"')):
        return text[1:-1].strip()
    return text

def _find_layer_in_current_project(layer_name):
    try:
        project = arcpy.mp.ArcGISProject("CURRENT")
    except Exception:
        return None

    target_name = (layer_name or "").strip().lower()
    if not target_name:
        return None

    for map_object in project.listMaps():
        for layer in map_object.listLayers():
            try:
                if layer.name and layer.name.strip().lower() == target_name:
                    return layer
            except Exception:
                continue

    return None

def _resolve_layer_reference(value):
    value = _strip_wrapping_quotes(value)
    if not value:
        return value

    try:
        if arcpy.Exists(value):
            return value
    except Exception:
        pass

    layer = _find_layer_in_current_project(value)
    if layer is None:
        return value

    try:
        return layer
    except Exception:
        pass

    try:
        data_source = layer.dataSource
        if data_source and arcpy.Exists(data_source):
            return data_source
    except Exception:
        pass

    return value

def _copy_selected(input_layer, output_fc):
    desc = arcpy.Describe(input_layer)
    arcpy.management.CopyFeatures(input_layer, output_fc)
    if hasattr(desc, "FIDSet") and desc.FIDSet:
        _msg(f"Copied {len(desc.FIDSet.split(';'))} selected parcel(s).")
    else:
        _msg("No selection on parcels; copied all features.")
    return output_fc

def _get_multivalue_parameter_inputs(param):
    raw_items = []

    try:
        values = getattr(param, "values", None)
        if values:
            for value in values:
                text = getattr(value, "valueAsText", None)
                if text is None:
                    text = str(value)
                text = (text or "").strip()

                if text and text not in ("#", "None", "[]"):
                    raw_items.append(text)
    except Exception:
        pass

    if not raw_items:
        text = (param.valueAsText or "").strip()
        if text and text not in ("#", "None", "[]"):
            raw_items = [
                s.strip()
                for s in text.split(";")
                if s.strip() and s.strip() not in ("#", "None", "[]")
            ]

    resolved_items = [_resolve_layer_reference(item) for item in raw_items]

    _msg(f"[Parcels] Parsed {len(raw_items)} item(s) from parameter; resolved to {len(resolved_items)} reference(s).")
    for i, (raw_item, resolved_item) in enumerate(zip(raw_items, resolved_items), start=1):
        if hasattr(resolved_item, "name"):
            try:
                _msg(f"  {i}) RAW={raw_item}  -> LAYER={resolved_item.name}")
            except Exception:
                _msg(f"  {i}) RAW={raw_item}  -> LAYER_OBJECT")
        else:
            _msg(f"  {i}) RAW={raw_item}  -> {resolved_item}")

    return resolved_items

def _validate_same_geometry(feature_list):
    if not feature_list:
        return

    base_shape_type = arcpy.Describe(feature_list[0]).shapeType
    for feature in feature_list[1:]:
        shape_type = arcpy.Describe(feature).shapeType
        if shape_type != base_shape_type:
            raise arcpy.ExecuteError(
                f"Parcel inputs geometry mismatch: {feature_list[0]} is {base_shape_type}"
                f"but {feature} is {shape_type}"
            )

def _copy_selected_multi(parcel_inputs, output_fc):
    if not parcel_inputs:
        raise arcpy.ExecuteError("Parcel Features is empty.")

    for parcel_input in parcel_inputs:
        if hasattr(parcel_input, "name"):
            continue
        if not arcpy.Exists(parcel_input):
            raise arcpy.ExecuteError(f"Parcel input does not exist: {parcel_input}")

    _validate_same_geometry(parcel_inputs)

    memory_workspace = "in_memory"
    temp_feature_classes = []

    for i, parcel_input in enumerate(parcel_inputs, start=1):
        temp_fc = _unique_name(f"parcels_sel_{i}", memory_workspace)

        if hasattr(parcel_input, "name"):
            _msg(f"[Parcels] Copying input {i}/{len(parcel_inputs)} (layer): {parcel_input.name}")
        else:
            _msg(f"[Parcels] Copying input {i}/{len(parcel_inputs)}: {parcel_input}")
        _copy_selected(parcel_input, temp_fc)
        temp_feature_classes.append(temp_fc)

    if len(temp_feature_classes) == 1:
        arcpy.management.CopyFeatures(temp_feature_classes[0], output_fc)
        _msg("[Parcels] Single input; no merge needed.")
    else:
        arcpy.management.Merge(temp_feature_classes, output_fc)
        _msg(f"[Parcels] Merged {len(temp_feature_classes)} copied inputs into one parcels FC.")

    return output_fc

def _collect_region(region_param_value, output_name):
    memory_workspace = "in_memory"
    output_fc = _unique_name(output_name, memory_workspace)

    if not region_param_value or str(region_param_value) in ("#", "", "[]", "None"):
        _warn(f"{output_name}: region is empty; nothing will be assigned to this split.")
        spatial_ref = arcpy.SpatialReference(4326)
        workspace = os.path.dirname(output_fc)
        name = os.path.basename(output_fc)
        arcpy.management.CreateFeatureclass(workspace, name, "POLYGON", spatial_reference=spatial_ref)
        return output_fc

    arcpy.management.CopyFeatures(region_param_value, output_fc)
    return output_fc

def _scan_labeled_root(labeled_root):
    labeled_root = _normalize_path(labeled_root)

    files_by_address = defaultdict(list)
    address_keys = {}
    category_counts = defaultdict(int)
    total_files = 0

    for category_name in os.listdir(labeled_root):
        category_path = os.path.join(labeled_root, category_name)

        if not os.path.isdir(category_path):
            continue

        if category_name.lower() not in CATEGORIES:
            _warn(f"Skipping non-category at root: {category_name}")
            continue

        for address_name in os.listdir(category_path):
            address_path = os.path.join(category_path, address_name)

            if not os.path.isdir(address_path):
                continue

            address_key = _canon_key(address_name)
            address_keys[(category_name, address_name)] = address_key

            for file_name in os.listdir(address_path):
                file_path = os.path.join(address_path, file_name)

                if not os.path.isfile(file_path):
                    continue

                extension = os.path.splitext(file_name)[1].lower()

                if extension in IMAGE_EXTENSIONS:
                    files_by_address[(category_name, address_name)].append(_normalize_path(file_path))
                    total_files += 1
                    category_counts[category_name] += 1

    _msg(f"[Scan] Found {total_files} frame(s) in labeled root across {len(files_by_address)} address folder(s).")

    if total_files == 0:
        _warn("No frames found. Check your Labeled Folder structure (Category/Address/Frames).")

    return files_by_address, address_keys

def _build_parcel_match_string(row_dict, own_field, pin_field, legal_field):
    own_value = row_dict.get(own_field)
    pin_value = row_dict.get(pin_field)
    legal_value = row_dict.get(legal_field)

    own_value = str(own_value).strip() if own_value not in (None, "", " ") else ""
    pin_value = str(pin_value).strip() if pin_value not in (None, "", " ") else ""
    legal_value = str(legal_value).strip() if legal_value not in (None, "", " ") else ""

    if own_value or pin_value:
        parts = []
        if own_value: parts.append(own_value)
        if pin_value: parts.append(pin_value)
        return " - ".join(parts)

    if legal_value:
        return legal_value

    return ""

def _oid_set_from_location(layer, relation, region_fc):
    layer_name = f"lyr_{uuid.uuid4().hex[:8]}"
    arcpy.management.MakeFeatureLayer(layer, layer_name)

    try:
        arcpy.management.SelectLayerByLocation(
            in_layer=layer_name,
            overlap_type=relation,
            select_features=region_fc,
            selection_type="NEW_SELECTION"
        )
        return {row[0] for row in arcpy.da.SearchCursor(layer_name, ["OID@"])}
    finally:
        try:
            arcpy.management.Delete(layer_name)
        except Exception:
            pass

def _best_fuzzy_match(query_key, candidates_dict, min_ratio=0.86):
    best_key = None
    best_ratio = 0.0

    for candidate_key in candidates_dict.keys():
        ratio = SequenceMatcher(None, query_key, candidate_key).ratio()
        if ratio > best_ratio:
            best_key = candidate_key
            best_ratio = ratio

    if best_ratio >= min_ratio:
        return best_key, best_ratio

    return None, 0.0

def updateMessages(self, params):
    pass

# -----------------------------------------------------
# Toolbox
# -----------------------------------------------------
class Toolbox(object):
    def __init__(self):
        self.label = "Split Datasets"
        self.alias = "SplitDatasets"
        self.tools = [SplitDatasets]

class SplitDatasets(object):
    def __init__(self):
        self.label = "Split Datasets"
        self.description = (
            "Split dataset geographically into training, validation, and test sets by polygons"
        )

    def getParameterInfo(self):
        p = []

        p.append(arcpy.Parameter(
            displayName="Labeled Data Folder",
            name="labeled_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        ))

        p1 = arcpy.Parameter(
            displayName="Parcel Features",
            name="parcel_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        p1.multiValue = True
        p.append(p1)

        p.append(arcpy.Parameter(
            displayName="Training Region",
            name="training_region",
            datatype="GPFeatureRecordSetLayer",
            parameterType="Required",
            direction="Input"
        ))

        p.append(arcpy.Parameter(
            displayName="Validation Region",
            name="validation_region",
            datatype="GPFeatureRecordSetLayer",
            parameterType="Optional",
            direction="Input"
        ))

        p.append(arcpy.Parameter(
            displayName="Test Region",
            name="test_region",
            datatype="GPFeatureRecordSetLayer",
            parameterType="Optional",
            direction="Input"
        ))

        p.append(arcpy.Parameter(
            displayName="Training Set Folder",
            name="train_out_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Output"
        ))

        p.append(arcpy.Parameter(
            displayName="Validation Set Folder",
            name="validation_out_folder",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Output"
        ))

        p.append(arcpy.Parameter(
            displayName="Test Set Folder",
            name="test_out_folder",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Output"
        ))

        return p

    def execute(self, parameters, messages):
        arcpy.env.overwriteOutput = True

        labeled_root = parameters[0].valueAsText
        parcel_features_param = parameters[1]

        train_region_value = parameters[2].value
        validation_region_value = parameters[3].value
        test_region_value = parameters[4].value

        train_out = parameters[5].valueAsText
        validation_out = parameters[6].valueAsText
        test_out     = parameters[7].valueAsText

        files_by_address, address_keys = _scan_labeled_root(labeled_root)

        memory_workspace = "in_memory"
        selected_parcels_fc = _unique_name("parcels_sel", memory_workspace)

        parcel_inputs = _get_multivalue_parameter_inputs(parcel_features_param)
        _copy_selected_multi(parcel_inputs, selected_parcels_fc)

        train_region_fc = _collect_region(train_region_value, "train_region")
        validation_region_fc  = _collect_region(validation_region_value, "validation_region")
        test_region_fc  = _collect_region(test_region_value, "test_region")

        train_parcels = _oid_set_from_location(selected_parcels_fc, "COMPLETELY_WITHIN", train_region_fc)
        validation_parcels  = _oid_set_from_location(selected_parcels_fc, "COMPLETELY_WITHIN", validation_region_fc)
        test_parcels  = _oid_set_from_location(selected_parcels_fc, "COMPLETELY_WITHIN", test_region_fc)

        _msg(
            f"Parcels in training region: {len(train_parcels)}"
            f"Parcel in validation region: {len(validation_parcels)}"
            f"Parcel in test region: {len(test_parcels)}"
        )

        own_address_field = _find_existing_field(
            selected_parcels_fc,
            ["Own_Addres", "Own_Address", "Owner_Address", "OWN_ADDRES"]
        )

        pin_field = _find_existing_field(
            selected_parcels_fc,
            ["PIN", "Parcel_ID", "PARCEL", "PID"]
        )

        legal_description_field = _find_existing_field(
            selected_parcels_fc,
            ["Legal Description", "LEGAL_DESCRIPTION", "Legal_Description", "LegalDesc", "LEG_DESC"]
        )

        if not (own_address_field or legal_description_field):
            _warn("Parcel fields for Own_Addres/PIN/Legal Description were not found; address matching may fail.")

        parcel_match_map = {}
        field_list = ["OID@", own_address_field, pin_field, legal_description_field]
        field_list = [field for field in field_list if field]

        with arcpy.da.SearchCursor(selected_parcels_fc, field_list) as cursor:
            for row in cursor   :
                row_dict = {}
                row_dict = {"OID@": row[0]}
                index = 1

                if own_address_field:
                    row_dict[own_address_field] = row[index]; index += 1
                if pin_field:
                    row_dict[pin_field] = row[index]; index += 1
                if legal_description_field:
                    row_dict[legal_description_field] = row[index]; index += 1

                display_string = _build_parcel_match_string(
                    row_dict,
                    own_address_field,
                    pin_field,
                    legal_description_field
                )

                parcel_key  = _canon_key(display_string)

                if not parcel_key:
                    continue

                split_name = "UNASSIGNED"
                oid_value = row_dict["OID@"]

                if oid_value in train_parcels:
                    split_name = "train"
                if oid_value in validation_parcels:
                    split_name = "validation"
                if oid_value in test_parcels:
                    split_name = "test"

                parcel_match_map[parcel_key] = {
                    "display": display_string,
                    "split": split_name,
                    "oid": oid_value,
                    "own": row_dict.get(own_address_field, ""),
                    "pin": row_dict.get(pin_field, ""),
                    "legal": row_dict.get(legal_description_field, "")
                }

        _msg(f"Built {len(parcel_match_map)} parcel match key(s).")

        _ensure_folder(train_out)
        _ensure_folder(validation_out)
        _ensure_folder(test_out)

        counts = defaultdict(int)
        unmatched = []
        matched_rows = 0
        copied_files = 0

        for (category_name, address_name), file_list in files_by_address.items():
                address_key = address_keys[(category_name, address_name)]

                match_method = "none"
                match_ratio = ""
                notes = ""

                split_name = "UNASSIGNED"
                display_string = ""
                parcel_oid = ""
                own_value = ""
                pin_value = ""
                legal_value = ""
                copied_count = 0

                if address_key in parcel_match_map:
                    info = parcel_match_map[address_key]
                    display_string = info["display"]
                    split_name = info["split"]
                    parcel_oid = info["oid"]
                    own_value = info["own"]
                    pin_value = info["pin"]
                    legal_value = info["legal"]
                    match_method = "exact"
                    match_ratio = "1.00"
                else:
                    best_key, best_ratio = _best_fuzzy_match(address_key, parcel_match_map, min_ratio=0.86)
                    if best_key:
                        info = parcel_match_map[best_key]
                        display_string = info["display"]
                        split_name = info["split"]
                        parcel_oid = info["oid"]
                        own_value = info["own"]
                        pin_value = info["pin"]
                        legal_value = info["legal"]
                        match_method = "fuzzy"
                        match_ratio = f"{best_ratio:.2f}"
                    else:
                        notes = "no_parcel_match"
                        unmatched.append((category_name, address_name, len(file_list)))
                        continue

                if split_name == "train":
                    destination_root = train_out
                elif split_name == "validation":
                    destination_root = validation_out
                elif split_name == "test":
                    destination_root = test_out
                else:
                    destination_root = None

                if destination_root:
                    destination_folder = os.path.join(destination_root, category_name, address_name)
                    _ensure_folder(destination_folder)

                    for src_file in file_list:
                        try:
                            dst_file = os.path.join(destination_folder, os.path.basename(src_file))
                            if not os.path.exists(dst_file):
                                shutil.copy2(src_file, dst_file)
                            copied_files += 1
                            counts[(split_name, category_name)] += 1
                        except Exception as e:
                            notes = (notes + f";copy_error:{e}").strip(";")
                else:
                    notes = (notes + ";parcel_unassigned_to_region").strip(";")

                matched_rows += 1
