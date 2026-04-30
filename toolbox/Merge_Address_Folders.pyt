import os, shutil, hashlib, fnmatch, csv
import arcpy

# -----------------------------------------------------
# Helpers functions
# -----------------------------------------------------
def _normalize_path(path):
    return os.path.normpath(os.path.abspath(path))

def _iter_matching_files(folder_path, patterns):
    try:
        for name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, name)

            if os.path.isfile(file_path):
                if not patterns:
                    yield file_path
                else:
                    for pattern in patterns:
                        if fnmatch.fnmatch(name, pattern):
                            yield file_path
                            break
    except FileNotFoundError:
        # folder does not exist
        return

def _sha1(path, buffer_size=1024*1024):
    hasher = hashlib.sha1()

    with open(path, 'rb') as f:
        while True:
            chunk = f.read(buffer_size)
            if not chunk: break
            hasher.update(chunk)
    return hasher.hexdigest()

def _safe_copy_files(src_path, dst_folder, overwrite_mode, use_hash_dedup, seen_hashes, dry_run):
    os.makedirs(dst_folder, exist_ok=True)

    base_name = os.path.basename(src_path)
    dst_path = os.path.join(dst_folder, base_name)

    # optional deduplication
    if use_hash_dedup:
        file_hash = _sha1(src_path)
        if file_hash in seen_hashes:
            return ("skipped_hashdup", dst_path)
    else:
        file_hash = None

    # destination already exists
    if os.path.exists(dst_path):
        if overwrite_mode == "overwrite":
            if not dry_run: shutil.copy2(src_path, dst_path)
            if use_hash_dedup and file_hash: seen_hashes.add(file_hash)
            return ("overwritten", dst_path)

        elif overwrite_mode == "skip":
            if use_hash_dedup and file_hash: seen_hashes.add(file_hash)
            return ("skipped_exists", dst_path)

        else:  # rename
            root, ext = os.path.splitext(base_name)
            copy_index = 1
            renamed_dst_path = os.path.join(dst_folder, f"{root} ({copy_index}){ext}")

            while os.path.exists(renamed_dst_path):
                copy_index += 1
                renamed_dst_path = os.path.join(dst_folder, f"{root} ({copy_index}){ext}")
            if not dry_run: shutil.copy2(src_path, renamed_dst_path)
            if use_hash_dedup and file_hash: seen_hashes.add(file_hash)
            return ("renamed", renamed_dst_path)

    else:
        if not dry_run: shutil.copy2(src_path, dst_path)
        if use_hash_dedup and file_hash: seen_hashes.add(file_hash)
        return ("copied", dst_path)

def _split_multivalue_text(text):
    if not text:
        return []

    parts = text.split(";")
    cleaned = []

    for part in parts:
        part = part.strip()
        if len(part) >= 2 and ((part[0] == part[-1] == "'") or (part[0] == part[-1] == '"')):
            part = part[1:-1]
        if part:
            cleaned.append(part)

    return cleaned

# =========================================================
# Toolbox
# =========================================================
class Toolbox(object):
    def __init__(self):
        self.label = "Merge Address Folders"
        self.alias = "MergeAddressFolders"
        self.tools = [MergeAddressFolders]

# =========================================================
# Main tool
# =========================================================
class MergeAddressFolders(object):
    def __init__(self):
        self.label = "Merge Address Folders"
        self.description = (
            "Merge multiple parent folders into one output. "
            "Surrounding subfolders merge together separately."
        )

        # Internal defaults
        self._SURROUNDING_FOLDER_NAME  = "Surrounding"
        self._FILE_PATTERNS  = ["*.jpg","*.jpeg","*.png","*.mp4"]
        self._OVERWRITE_MODE = "rename"   # skip, rename, or overwrite
        self._USE_HASH_DEDUP  = True
        self._DRY_RUN = False
        self._MANIFEST_CSV = None

    def getParameterInfo(self):
        p = []

        in_roots = arcpy.Parameter(
            displayName="Input Address Folders",
            name="in_roots",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        p.append(in_roots)

        out_root = arcpy.Parameter(
            displayName="Output Address Folders",
            name="out_root",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )
        p.append(out_root)

        return p

    def updateMessages(self, params):
        # Parse multi-value as semicolon-separated text
        in_roots = _split_multivalue_text(params[0].valueAsText)
        out_root = params[1].valueAsText

        if not in_roots:
            params[0].setErrorMessage("Provide at least one input address folder.")

        if out_root:
            try:
                out_root_abs = _normalize_path(out_root)
            except Exception:
                out_root_abs = None

            for in_root in in_roots:
                try:
                    in_root_abs = _normalize_path(in_root)

                    # prevent output being nested inside any input
                    if out_root_abs and out_root_abs.startswith(in_root_abs + os.sep):
                        params[1].setErrorMessage(
                        "Output folder cannot be inside an input folder."
                        )
                        break
                except Exception:
                    pass

    def execute(self, params, messages):
        in_roots = _split_multivalue_text(params[0].valueAsText)
        out_root = _normalize_path(params[1].valueAsText)

        surrounding_name = self._SURROUNDING_FOLDER_NAME
        surrounding_lower = surrounding_name.lower()
        file_patterns = list(self._FILE_PATTERNS)
        overwrite_mode = self._OVERWRITE_MODE
        dedup_hash = bool(self._USE_HASH_DEDUP)
        dry_run = bool(self._DRY_RUN)
        manifest_csv = self._MANIFEST_CSV

        os.makedirs(out_root, exist_ok=True)

        # Optional manifest log
        manifest_writer = None
        manifest_file = None
        if manifest_csv:
            manifest_file = open(manifest_csv, "w", newline="", encoding="utf-8")
            manifest_writer = csv.writer(manifest_file)
            manifest_writer.writerow(["source_path", "target_path", "bucket", "action"])

        # track hashes separately for each address / bucket
        seen_hashes_by_bucket = {}
        totals = {
            "copied":0,
            "overwritten":0,
            "renamed":0,
            "skipped_exists":0,
            "skipped_hashdup":0
        }

        def log_action(src_path, dst_path, bucket_name, action):
            totals[action] = totals.get(action, 0) + 1
            if manifest_writer: manifest_writer.writerow([src_path, dst_path, bucket_name, action])


        # ------------------------------------------------------------------
        # Discover address folders across all input roots
        # ------------------------------------------------------------------
        all_address_names = set()
        address_dirs_by_root = []

        for in_root in in_roots:
            try:
                in_root_abs = _normalize_path(in_root)
            except Exception:
                arcpy.AddWarning(f"Unreadable input: {in_root}")
                address_dirs_by_root.append({})
                continue

            if not os.path.isdir(in_root_abs):
                arcpy.AddWarning(f"Input not a folder or unreadable: {in_root}")
                address_dirs_by_root.append({})
                continue

            address_map = {}

            try:
                for name in os.listdir(in_root_abs):
                    path = os.path.join(in_root_abs, name)

                    if os.path.isdir(path):
                        address_map[name] = path
                        all_address_names.add(name)

            except PermissionError:
                arcpy.AddWarning(f"Permission denied: {in_root_abs}")

            address_dirs_by_root.append(address_map)

        if not all_address_names:
            arcpy.AddWarning("No address folders found in the provided inputs.")
            if manifest_file: manifest_file.close()
            return

        arcpy.AddMessage(f"Discovered {len(all_address_names)} unique address folders.")

        # ------------------------------------------------------------------
        # Merge one address at a time
        # ------------------------------------------------------------------
        for address_name in sorted(all_address_names):
            output_address_folder = os.path.join(out_root, address_name)
            output_main_folder = output_address_folder
            output_surrounding_folder = os.path.join(output_address_folder, surrounding_name)

            # separate hash tracking for main vs. surrounding
            seen_hashes_by_bucket.setdefault((address_name, "main"), set())
            seen_hashes_by_bucket.setdefault((address_name, "surrounding"), set())

            main_files_added = 0
            surrounding_files_added = 0

            for address_map in address_dirs_by_root:
                input_address_folder = address_map.get(address_name)
                if not input_address_folder:
                    continue

                # ----------------------------------------------------------
                # Copy top-level files from the address folder
                # ----------------------------------------------------------
                for src_file_path in _iter_matching_files(input_address_folder, file_patterns):
                    action, dst_path = _safe_copy_files(
                        src_file_path,
                        output_main_folder,
                        overwrite_mode,
                        dedup_hash,
                        seen_hashes_by_bucket[(address_name, "main")],
                        dry_run
                    )

                    log_action(src_file_path, dst_path, "main", action)

                    if action in ("copied", "overwritten", "renamed"):
                        main_files_added += 1

                # ----------------------------------------------------------
                # Copy files from the Surrounding subfolder
                # ----------------------------------------------------------
                try:
                    for child_name in os.listdir(input_address_folder):
                        child_path = os.path.join(input_address_folder, child_name)

                        if os.path.isdir(child_path) and child_name.lower() == surrounding_lower:
                            for src_file_path in _iter_matching_files(child_path, file_patterns):
                                action, dst_path = _safe_copy_files(
                                    src_file_path,
                                    output_surrounding_folder,
                                    overwrite_mode,
                                    dedup_hash,
                                    seen_hashes_by_bucket[(address_name, "surrounding")],
                                    dry_run
                                )

                                log_action(src_file_path, dst_path, "surrounding", action)

                                if action in ("copied", "overwritten", "renamed"):
                                    surrounding_files_added += 1

                except PermissionError:
                    arcpy.AddWarning(f"Permission denied reading: {input_address_folder}")
                except Exception as e:
                    arcpy.AddWarning(f"Error reading '{input_address_folder}': {e}")

            arcpy.AddMessage(f"[{address_name}] main +{main_files_added} | surrounding +{surrounding_files_added}")
