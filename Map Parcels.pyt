import os, re, time, uuid
from pathlib import Path
import arcpy

# -------------------------------------------------
# Hard-code recovery states
# -------------------------------------------------
LABELS = ["Other", "Uninhabited", "Empty", "Rebuilding", "Rebuilt"]

LABEL_SCORE = {
    "OTHER":        0,
    "UNINHABITED":  1,
    "EMPTY":        2,
    "REBUILDING":   3,
    "REBUILT":      4,
}

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def _snap_label(folder_name):
    """Snap any folder name to one of defined recovery states."""
    clean_name = re.sub(r"[^A-Z]+", " ", (folder_name or "").upper()).strip()

    for label in LABELS:
        if clean_name == label.upper():
            return label

    return "Other"

def _label_score(label):
    return LABEL_SCORE.get((label or "Other").upper(), 0)

def _normalize_text(value):
    if value is None:
        return ""

    value = re.sub(r"[^A-Z0-9]+", " ", str(value).upper())
    return re.sub(r"\s+", " ", value).strip()

def _find_field_case_insensitive(feature_class, wanted_name):
    wanted_key = re.sub(r"[\s_]+","", wanted_name).lower()

    for field in arcpy.ListFields(feature_class) or []:
        field_key = re.sub(r"[\s_]+","", field.name).lower()
        if field_key == wanted_key:
            return field.name

    return None

def _safe_fc_base_name(name):
    return re.sub(r'[^A-Za-z0-9_]+', '_', os.path.basename(name))[:120]

def _safe_fc_name(name, gdb_path):
    try:
        return arcpy.ValidateTableName(os.path.basename(name), gdb_path)
    except Exception:
        return re.sub(r'[^A-Za-z0-9_]+', '_', os.path.basename(name))[:120]

def _wait_for_exists(path, tries=20, delay=0.2):
    for _ in range(tries):
        if arcpy.Exists(path):
            return True
        time.sleep(delay)

    return arcpy.Exists(path)

def _clear_workspace_locks():
    try:
        arcpy.ClearWorkspaceCache_management()
    except Exception:
        pass

    time.sleep(0.15)

def _trim_output_fields(fc):
    keep_names = {"OBJECTID", "PIN", "Own_Addres", "Label", "LabelScore", "GlobalID"}
    delete_fields = []

    for field in arcpy.ListFields(fc):
        # keep required/system fields and the fields we want
        if field.required or field.name in keep_names:
            continue
        delete_fields.append(field.name)

    if delete_fields:
        arcpy.management.DeleteField(fc, delete_fields)

def scan_labeled_folder(root_folders, exts=(".jpg", ".jpeg", ".png", ".mp4")):
    valid_exts = {e.lower() for e in exts}
    hits_by_address = {}

    for root_folder in root_folders:
        for top_label_name in os.listdir(root_folder):
            label_folder = os.path.join(root_folder, top_label_name)

            if not os.path.isdir(label_folder):
                continue

            snapped_label = _snap_label(top_label_name)

            for address_folder_name in os.listdir(label_folder):
                address_folder = os.path.join(label_folder, address_folder_name)
                if not os.path.isdir(address_folder):
                    continue

                normalized_key = _normalize_text(address_folder_name)
                bucket = hits_by_address.setdefault(
                    normalized_key,
                    {"labels": set(), "frames": []}
                )
                bucket["labels"].add(snapped_label)

                for current_root, dirs, files in os.walk(address_folder):
                    # skip the "Surrounding" folder entirely ----
                    dirs[:] = [d for d in dirs if d.lower() != "surrounding"]

                    for filename in files:
                        if os.path.splitext(filename)[1].lower() in valid_exts:
                            bucket["frames"].append(os.path.join(current_root, filename))

    return hits_by_address

def _make_local_scratch_gdb():
    base_dir = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    run_root = os.path.join(
        base_dir,
        "SimpleAttach_Temp",
        f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    os.makedirs(run_root, exist_ok=True)

    gdb_path = os.path.join(run_root, "scratch.gdb")
    arcpy.management.CreateFileGDB(run_root, "scratch")
    return gdb_path

def _copy_back_if_possible(temp_fc_path, desired_fc_path, retries=6, delay=1.25):
    if os.path.normpath(temp_fc_path) == os.path.normpath(desired_fc_path):
        return temp_fc_path

    target_gdb  = os.path.dirname(desired_fc_path)
    target_name = _safe_fc_name(desired_fc_path, target_gdb)
    desired_fixed_path = os.path.join(target_gdb, target_name)

    for i in range(retries):
        try:
            if arcpy.Exists(desired_fixed_path):
                arcpy.management.Delete(desired_fixed_path)

            try:
                arcpy.management.CopyFeatures(temp_fc_path, desired_fixed_path)
            except Exception:
                arcpy.conversion.FeatureClassToFeatureClass(
                    temp_fc_path,
                    target_gdb,
                    target_name,
                )

            if _wait_for_exists(desired_fixed_path):
                arcpy.AddMessage(f"[CopyBack] Wrote to target: {desired_fixed_path}")
                return desired_fixed_path

        except Exception as e:
            arcpy.AddWarning(f"[CopyBack {i+1}/{retries}] {e}")

        time.sleep(delay)

    arcpy.AddWarning(f"[CopyBack] Target stayed locked; keeping result at {temp_fc_path}")
    return temp_fc_path

# -------------------------------------------------
# Toolbox
# -------------------------------------------------
class Toolbox(object):
    def __init__(self):
        self.label = "Map Parcel"
        self.alias = "MapParcel"
        self.tools = [MapParcel]

# -------------------------------------------------
# Main Toolbox
# -------------------------------------------------
class MapParcel(object):
    def __init__(self):
        self.label = "Map Parcels"
        self.description = "Map parcels using labeled data."

    def getParameterInfo(self):
        p0 = arcpy.Parameter(
            displayName="Labeled Address Folders",
            name="labeled_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )
        p0.multiValue = True

        p1 = arcpy.Parameter(
            displayName="Parcels",
            name="in_parcels",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        p1.filter.list = ["Polygon"]
        p1.multiValue = True

        p2 = arcpy.Parameter(
            displayName="Parcels with Labels",
            name="out_fc",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output"
        )
        return [p0, p1, p2]

    def updateMessages(self, params):
        labeled_folders = params[0].valueAsText
        output_fc = params[2].valueAsText
        if labeled_folders:
            folder_list = [
                s.strip().strip('"').strip("'")
                for s in labeled_folders.split(";")
                if s.strip()
        ]
        bad_folders = [folder for folder in folder_list if not os.path.isdir(folder)]
        if bad_folders:
            params[0].setErrorMessage("Pick an existing folder. Invalid: {bad_folders[0]}")

        if output_fc and (not os.path.dirname(output_fc).lower().endswith(".gdb")):
            params[2].setErrorMessage("Output must be a feature class inside a File GDB.")

    def execute(self, params, messages):
        arcpy.env.addOutputsToMap = False

        labeled_roots = []
        raw_text = params[0].valueAsText or ""

        for piece in raw_text.split(";"):
            piece = piece.strip().strip('"').strip("'")
            if piece:
                labeled_roots.append(piece)

        # multi-value feature-layer input
        raw_parcel_values = params[1].values
        if not raw_parcel_values:
            raise arcpy.ExecuteError("No parcels provided.")

        parcel_sources = []
        for value in raw_parcel_values:
            if value is not None:
                parcel_sources.append(value)

        if not parcel_sources:
            raise arcpy.ExecuteError("No valid parcel layers found in input.")

        # if multiple parcel layers were given, merge them first
        if len(parcel_sources) == 1:
            source_fc = parcel_sources[0]
            arcpy.AddMessage(f"[Input] Using single parcels layer: {source_fc}")
        else:
            merge_gdb = _make_local_scratch_gdb()
            source_fc = os.path.join(merge_gdb, "MergedParcels")

            arcpy.AddMessage(
                f"[Input] Merging {len(parcel_sources)} parcel layers into temporary FC:\n  {source_fc}"
            )

            for i, src in enumerate(parcel_sources, 1):
                arcpy.AddMessage(f"  [{i}] {src}")

            arcpy.management.Merge(parcel_sources, source_fc)

            if not _wait_for_exists(source_fc):
                raise arcpy.ExecuteError(
                    "Failed to merge input parcel layers into a temporary feature class."
                )

        desired_output_fc = params[2].valueAsText
        target_gdb = os.path.dirname(desired_output_fc)
        base_name  = _safe_fc_base_name(os.path.basename(desired_output_fc))

        output_fc = os.path.join(target_gdb, base_name)

        # try writing directly to the requested output first
        if arcpy.Exists(output_fc):
            arcpy.management.Delete(output_fc)

        created = False

        try:
            arcpy.management.CopyFeatures(source_fc, output_fc)
            created = _wait_for_exists(output_fc)
        except Exception:
            created = False

        if not created:
            try:
                arcpy.conversion.FeatureClassToFeatureClass(source_fc, target_gdb, base_name)
                created = _verify_exists(output_fc)
            except Exception:
                created = False

        # Fallback to local scratch if target gdb is locked
        if not created:
            scratch_gdb = _make_local_scratch_gdb()
            output_fc = os.path.join(scratch_gdb, base_name)

            arcpy.AddWarning(f"Output workspace locked. Writing to local scratch: {out_fc}")

            try:
                arcpy.management.CopyFeatures(source_fc, output_fc)
            except Exception:
                arcpy.conversion.FeatureClassToFeatureClass(source_fc, scratch_gdb, base_name)

            if not _wait_for_exists(output_fc):
                raise arcpy.ExecuteError(
                    "Could not create the output feature class in either location."
                )

        # ensure the required fields exist
        existing_fields = {f.name.lower(): f for f in (arcpy.ListFields(output_fc) or [])}

        if "label" not in existing_fields:
            arcpy.management.AddField(output_fc, "Label", "TEXT", field_length=32, field_is_nullable="NULLABLE")
        if "labelscore" not in existing_fields:
            arcpy.management.AddField(output_fc, "LabelScore", "SHORT")

        # resolve fields to match addresses
        oid_field = arcpy.Describe(output_fc).OIDFieldName
        address_field  = (
            _find_field_case_insensitive(output_fc, "Own_Addres")
            or _find_field_case_insensitive(output_fc, "Address")
        )
        pin_field   = _find_field_case_insensitive(output_fc, "PIN")

        if not address_field:
            raise arcpy.ExecuteError("Could not find an address field (e.g., 'Own_Addres' or 'Address') on the parcels.")

        # scan labeled folder once
        folder_hits = scan_labeled_folder(labeled_roots)
        arcpy.AddMessage(f"[Scan] {len(folder_hits)} address keys found in labeled folder.")

        fields = [oid_field, address_field] + ([pin_field] if pin_field else []) + [
            "Label",
            "LabelScore",
        ]
        field_index = {name:i for i, name in enumerate(fields)}

        updated_count = 0

        with arcpy.da.UpdateCursor(output_fc, fields) as cursor:
            for row in cursor:
                address_value = row[field_index[address_field]]
                pin_value  = row[field_index[pin_field]] if pin_field else None

                key_addr_pin = _normalize_text(f"{address_value} {pin_value}") if (address_value and pin_value) else None
                key_addr_only = _normalize_text(address_value) if address_value else None

                match = folder_hits.get(key_addr_pin) if key_addr_pin else None
                if not match:
                    match = folder_hits.get(key_addr_only) if key_addr_only else None

                if match:
                    best_label = max(match["labels"], key=lambda label: _label_score(label))

                    row[field_index["Label"]] = best_label
                    row[field_index["LabelScore"]] = _label_score(best_label)
                else:
                    row[field_index["Label"]] = None
                    row[field_index["LabelScore"]] = -1

                cursor.updateRow(row)
                updated_count += 1

        arcpy.AddMessage(
            f"[Labels] Updated {updated_count} parcels from folder labels.")

        _trim_output_fields(output_fc)

        # auto-copy back to the user's chosen target
        final_output_fc = _copy_back_if_possible(output_fc, desired_output_fc)

        params[2].value = final_output_fc
        arcpy.AddMessage(f"[Completed] Output feature class: {final_output_fc}")