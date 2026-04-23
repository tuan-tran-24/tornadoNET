import os, sys, math, json, shutil, re, time
from pathlib import Path
from collections import defaultdict
import arcpy, pandas as pd, numpy as np

# =========================================================
# Toolbox
# =========================================================
class Toolbox(object):
    def __init__(self):
        self.label = "Process Spatial Video & GIS"
        self.alias = "ProcessSpatialVideo&GIS"
        self.tools = [ProcessSpatialVideoGIS]

# =========================================================
# Main tool
# =========================================================
class ProcessSpatialVideoGIS(object):
    def __init__(self):
        self.label = "Process Spatial Video & GIS"
        self.description = (
            "Transform GIS and video frames to parcels, attach each GIS with corresponding "
            "frame, and sort images into address folders with 'Surrounding' folder."
        )

    # -----------------------------------------------------
    # parameters
    # -----------------------------------------------------
    def getParameterInfo(self):
        parameters = []

        gis_file = arcpy.Parameter(
            displayName="GIS File",
            name="gis_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )
        gis_file.filter.list = ["csv","json"]
        gis_file.description = "Files must be .csv or .json"
        parameters.append(gis_file)

        frames_folder = arcpy.Parameter(
            displayName="Frames Folder",
            name="frames_dir",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input"
        )
        parameters.append(frames_folder)

        camera_side = arcpy.Parameter(
            displayName="Camera Side",
            name="camera_side",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        camera_side.filter.type = "ValueList"
        camera_side.filter.list = ["Left", "Right"]
        camera_side.value = "Left"
        parameters.append(camera_side)

        gis_offset = arcpy.Parameter(
            displayName="GIS Offset",
            name="gis_offset",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        gis_offset.value = 0
        parameters.append(gis_offset)

        frame_offset = arcpy.Parameter(
            displayName="Frame Offset",
            name="frame_offset",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        frame_offset.value = 0
        parameters.append(frame_offset)

        parcels_fc = arcpy.Parameter(
            displayName="Parcel Features",
            name="parcels_fc",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        parcels_fc.filter.list = ["Polygon"]
        parameters.append(parcels_fc)

        houses_fc = arcpy.Parameter(
            displayName="House Features",
            name="houses_fc",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input"
        )
        houses_fc.filter.list = ["Polygon"]
        parameters.append(houses_fc)

        address_folders = arcpy.Parameter(
            displayName="Output Addresses Folder",
            name="base_dir",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input"
        )
        parameters.append(address_folders)

        attach_frames = arcpy.Parameter(
            displayName="Attach Frames",
            name="attach_frames",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        attach_frames.value = True
        parameters.append(attach_frames)

        output_points = arcpy.Parameter(
            displayName="Output Points",
            name="out_points",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output"
        )
        parameters.append(output_points)

        return parameters

    def updateParameters(self, parameters):
        gis_file        = parameters[0]
        output_points   = parameters[9]


        # -----------------------------------------------------
        # Helpers functions
        # -----------------------------------------------------
        def _pick_default_gdb():
            try:
                default_gdb = arcpy.mp.ArcGISProject("CURRENT").defaultGeodatabase
                if default_gdb and os.path.isdir(default_gdb): return default_gdb
            except Exception: pass

            try:
                workspace = arcpy.env.workspace
                if isinstance(workspace, str) and workspace.lower().endswith(".gdb") and os.path.isdir(workspace):
                    return workspace
            except Exception: pass

            try:
                if arcpy.env.scratchGDB and os.path.isdir(arcpy.env.scratchGDB):
                    return arcpy.env.scratchGDB
            except Exception: pass
            return None

        def _basename_from_gis():
            gis_path = gis_file.valueAsText or ""
            if not gis_path: return None
            base_name = os.path.splitext(os.path.basename(gis_path))[0]
            return arcpy.ValidateTableName(base_name)

        # If user hasn’t set an explicit output, auto-fill from GIS filename
        if (not output_points.altered) or not output_points.valueAsText:
            default_gdb = _pick_default_gdb()
            base_name   = _basename_from_gis()
            output_name = base_name

            if default_gdb:
                output_points.value = os.path.join(default_gdb, output_name)
            else:
                output_points.value = output_name
            return

        # If user typed a raw name, put it in a sensible GDB
        parent_folder = os.path.dirname(output_points.valueAsText or "")
        if ".gdb" not in (parent_folder or "").lower():
            default_gdb = _pick_default_gdb()
            if default_gdb:
                output_name = os.path.basename(output_points.valueAsText) or (
                    f"{_basename_from_gis() or 'Output'}_Points_Attached"
                )
                output_points.value = os.path.join(
                    default_gdb,
                    arcpy.ValidateTableName(output_name),
                )
        return

    def _catalog_paths_from_gpfeature_param(self, param, required=True, messages=None):
        def _resolve_one_path(raw_value):
            raw_value = str(raw_value).strip().strip("'").strip('"')
            desc = arcpy.Describe(raw_value)
            catalog_path = getattr(desc, "catalogPath", None) or raw_value
            if not arcpy.Exists(catalog_path):
                raise OSError(f"Not found: {raw_value}")
            return catalog_path

        resolved_paths = []

        param_value = getattr(param, "value", None)
        if param_value is not None and hasattr(param_value, "rowCount"):
            for row_index in range(param_value.rowCount):
                try:
                    resolved_paths.append(_resolve_one_path(param_value.getValue(row_index, 0)))
                except Exception as e:
                    if messages: messages.addWarningMessage(f"Skipping input: {e}")

        if not resolved_paths:
            text = (param.valueAsText or "").strip()
            if text:
                for piece in [s for s in text.split(";") if s.strip()]:
                    try:
                        resolved_paths.append(_resolve_one_path(piece))
                    except Exception as e:
                        if messages: messages.addWarningMessage(f"Skipping input: {e}")

        # de-duplicate to preserve order
        seen, unique_paths = set(), []
        for path in resolved_paths:
            if path not in seen:
                unique_paths.append(path)
                seen.add(path)

        if required and not unique_paths:
            raise arcpy.ExecuteError("No valid feature class or layer found for the parameter.")

        return unique_paths

    # -----------------------------------------------------
    # Execute
    # -----------------------------------------------------
    def execute(self, parameters, messages):
        start_time = time.time()
        arcpy.env.overwriteOutput = True
        arcpy.env.parallelProcessingFactor = "100%"

        # -------------------------------------------------
        # Read inputs
        # -------------------------------------------------
        gis_input             = parameters[0].valueAsText or ""
        frames_folder         = (parameters[1].valueAsText or "").strip()
        camera_side           = (parameters[2].valueAsText or "Left").strip()
        gis_offset            = int(parameters[3].value or 0)
        frame_start_id        = int(parameters[4].value or 0)
        address_folders       = (parameters[7].valueAsText or "").strip()
        attach_frames         = True if parameters[8].value is None else bool(parameters[8].value)
        output_feature_class  = parameters[9].valueAsText

        # Resolve parcel/house inputs to catalog paths
        parcel_paths = self._catalog_paths_from_gpfeature_param(
            parameters[5],
            required=True,
            messages=messages,
        )
        house_paths  = self._catalog_paths_from_gpfeature_param(
            parameters[6],
            required=False,
            messages=messages,
        )
        house_path   = house_paths[0] if house_paths else None

        # Refresh address folders
        refresh_folders = True

        if not gis_input:
            raise arcpy.ExecuteError("Provide a GIS File in .csv or .json format.")

        working_spatial_ref = arcpy.Describe(parcel_paths[0]).spatialReference
        if working_spatial_ref is None or working_spatial_ref.name == "Unknown":
            raise arcpy.ExecuteError("First parcel layer has no valid spatial reference; use a projected CRS.")

        wgs84 = arcpy.SpatialReference(4326)

        # -------------------------------------------------
        # Read GIS file and prepare point list
        # -------------------------------------------------
        read_start_time = time.time()
        file_ext = os.path.splitext(gis_input)[1].lower()

        if file_ext == ".csv":
            gis_df = pd.read_csv(gis_input)
            columns = {col.lower(): col for col in gis_df.columns}
            lat = columns.get("lat")
            lon = columns.get("lon")

            if not (lat and lon):
                raise arcpy.ExecuteError("CSV must contain 'lat' and 'lon' columns.")

            gis_df = (gis_df[[lat, lon]].rename(columns={lat:"lat", lon:"lon"})
                                .apply(pd.to_numeric, errors="coerce")).dropna().reset_index(drop=True)

        elif file_ext == ".json":
            with open(gis_input, "r", encoding="utf-8-sig") as f:
                json_data = json.load(f)

            parsed_points = []
            for item in (json_data.get("points") if isinstance(json_data, dict) else json_data):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    parsed_points.append((item[0], item[1]))  # lat, lon
                elif isinstance(item, dict) and "lat" in item and "lon" in item:
                    parsed_points.append((item["lat"], item["lon"]))

            if not parsed_points:
                raise arcpy.ExecuteError("No valid points were parsed from the JSON file.")

            gis_df = (
                pd.DataFrame(parsed_points, columns=["lat","lon"])
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
                .reset_index(drop=True)
            )

        else:
            raise arcpy.ExecuteError("GIS File must be .csv or .json.")

        arcpy.AddMessage(f"[Load coordinates] {len(gis_df)} point(s) read in {time.time()-read_start_time:.2f}s")

        if gis_offset > 0:
            if gis_offset >= len(gis_df):
                messages.addWarningMessage(
                    f"GIS Offset ({gis_offset}) ≥ total points ({len(gis_df)}). Output will be empty."
                )
                gis_df = gis_df.iloc[0:0].reset_index(drop=True)
            else:
                gis_df = gis_df.iloc[gis_offset:].reset_index(drop=True)
                arcpy.AddMessage(f"GIS Offset applied: dropped {gis_offset} point(s); {len(gis_df)} remain.")

        # -------------------------------------------------
        # Create output feature class
        # -------------------------------------------------
        output_container = os.path.dirname(output_feature_class)
        output_name = os.path.basename(output_feature_class)

        if not output_container or ".gdb" not in output_container.lower():
            raise arcpy.ExecuteError("Output must be a feature class inside a file geodatabase.")

        if arcpy.Exists(output_feature_class): arcpy.management.Delete(output_feature_class)

        arcpy.management.CreateFeatureclass(
            output_container,
            output_name,
            "POINT",
            spatial_reference=working_spatial_ref)

        output_fields = [
            ("SeqID", "LONG", {}),
            ("FrameID", "LONG", {}),
            ("ImagePath", "TEXT", {"field_length": 500, "field_alias": "Image Path"}),
            ("ParcelAddress", "TEXT", {"field_length": 255, "field_alias": "Parcel Address"}),
            ("IsSurrounding", "SHORT", {"field_alias": "IsSurrounding"}),
        ]

        for field_name, field_type, kwargs in output_fields:
            arcpy.management.AddField(output_feature_class, field_name, field_type, **kwargs)

        if gis_df.empty:
            parameters[9].value = output_feature_class
            arcpy.AddMessage("No points after GIS Offset; created empty output.")
            return

        # -------------------------------------------------
        # Bearings and projected source points
        # -------------------------------------------------
        lat_rad = np.deg2rad(pd.to_numeric(gis_df["lat"], errors="coerce").to_numpy())
        lon_rad = np.deg2rad(pd.to_numeric(gis_df["lon"], errors="coerce").to_numpy())

        delta_lon = (np.diff(lon_rad, prepend=lon_rad[0]) + np.pi) % (2*np.pi) - np.pi
        y = np.sin(delta_lon) * np.cos(lat_rad)
        x_step = (
            np.cos(lat_rad[:-1]) * np.sin(lat_rad[1:])
            - np.sin(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.cos(delta_lon[1:])
        )
        x = np.insert(x_step, 0, x_step[0] if x_step.size else 0.0)

        bearings = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0

        source_points = [
            arcpy.PointGeometry(arcpy.Point(float(lon), float(lat)), wgs84)
            .projectAs(working_spatial_ref)
            .firstPoint
            for lat, lon in zip(gis_df["lat"].to_numpy(), gis_df["lon"].to_numpy())
        ]

        # -------------------------------------------------
        # Unit conversion and constants
        # -------------------------------------------------
        meters_per_unit = working_spatial_ref.metersPerUnit or 1.0
        units_to_feet = meters_per_unit / 0.3048
        feet_to_units = 1.0 / units_to_feet

        # Transform constants
        hit_window_ft = 100.0
        long_probe_ft = 1500.0
        threshold_ft = 30.0

        hit_window_units = hit_window_ft * feet_to_units
        long_probe_units = long_probe_ft * feet_to_units

        # Carving constants
        carve_half_ft = 10.0
        ray_length_ft = 150.0
        d2_threshold_ft = 30.0
        d2_offset_ft = 10.0

        carve_half_units = carve_half_ft / units_to_feet
        ray_length_units = ray_length_ft / units_to_feet
        d2_threshold_units = d2_threshold_ft / units_to_feet
        d2_offset_units = d2_offset_ft / units_to_feet

        fan_radius_ft = 150.0
        fan_half_deg = 30.0
        fan_radius_units = fan_radius_ft * feet_to_units

        # -------------------------------------------------
        # Frame mapping
        # -------------------------------------------------
        extensions = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
        frame_number_pattern  = re.compile(r"_frame_(\d+)\.[^.]+$", re.IGNORECASE)

        frame_id_to_path = {}
        frames_available = bool(frames_folder) and os.path.isdir(frames_folder)

        if frames_available:
            with os.scandir(frames_folder) as scan_iter:
                for entry in scan_iter:
                    if entry.is_file():
                        _, ext = os.path.splitext(entry.name)
                        if ext.lower() in extensions:
                            match_frame = frame_number_pattern.search(entry.name)
                            if match_frame:
                                frame_id_to_path[int(match_frame.group(1))] = entry.path
        else:
            if attach_frames:
                arcpy.AddWarning(
                    "Attach Frames checked, but Frames Folder was not provided. Attachments will be skipped."
                )
                attach_frames = False

            if address_folders:
                arcpy.AddWarning(
                    "Output Addresses Folder was provided, but Frames Folder was not provided."
                    "Creating address folders will be skipped."
                )
        # -------------------------------------------------
        # Build route and ROI
        # -------------------------------------------------
        route_array = arcpy.Array([point for point in source_points])

        route_fc = os.path.join("in_memory","route_line")

        if arcpy.Exists(route_fc): arcpy.management.Delete(route_fc)
        arcpy.management.CopyFeatures([arcpy.Polyline(route_array, working_spatial_ref)], route_fc)

        roi_buffer_ft = 400.0
        roi_fc = os.path.join("in_memory","route_roi")
        if arcpy.Exists(roi_fc): arcpy.management.Delete(roi_fc)
        arcpy.analysis.Buffer(route_fc, roi_fc, f"{roi_buffer_ft} Feet", "FULL", "ROUND", "ALL")

        # -------------------------------------------------
        # Prepare parcel geometries
        # -------------------------------------------------
        merged_parcels_fc = os.path.join("in_memory", "parcels_mrg")
        if arcpy.Exists(merged_parcels_fc): arcpy.management.Delete(merged_parcels_fc)

        if len(parcel_paths) == 1:
            arcpy.management.CopyFeatures(parcel_paths[0], merged_parcels_fc)
        else:
            arcpy.management.Merge(parcel_paths, merged_parcels_fc)

        parcels_roi_fc = os.path.join("in_memory","parcels_roi")
        if arcpy.Exists(parcels_roi_fc): arcpy.management.Delete(parcels_roi_fc)

        arcpy.management.MakeFeatureLayer(merged_parcels_fc, "parcels_mrg_fl")
        arcpy.management.SelectLayerByLocation(
            "parcels_mrg_fl",
            "INTERSECT",
            roi_fc,
            selection_type="NEW_SELECTION",
        )

        if int(arcpy.management.GetCount("parcels_mrg_fl")[0]) > 0:
            arcpy.management.CopyFeatures("parcels_mrg_fl", parcels_roi_fc)
        else:
            arcpy.management.CopyFeatures(merged_parcels_fc, parcels_roi_fc)

        projected_points_fc = os.path.join("in_memory", "pts_tmp_pj")
        if arcpy.Exists(projected_points_fc): arcpy.management.Delete(projected_points_fc)

        arcpy.management.CreateFeatureclass(
            "in_memory",
            "pts_tmp_pj",
            "POINT",
            spatial_reference=working_spatial_ref,
        )
        arcpy.management.AddField(projected_points_fc, "IDX", "LONG")
        arcpy.management.AddField(projected_points_fc, "GID", "SHORT")

        with arcpy.da.InsertCursor(projected_points_fc, ["SHAPE@", "IDX", "GID"]) as insert_cursor:
            for point_index, source_point in enumerate(source_points):
                insert_cursor.insertRow([arcpy.PointGeometry(source_point, working_spatial_ref), point_index, 1])

        route_buffer_fc = os.path.join("in_memory", "route_buf")
        if arcpy.Exists(route_buffer_fc): arcpy.management.Delete(route_buffer_fc)
        arcpy.analysis.Buffer(
            route_fc,
            route_buffer_fc,
            f"{carve_half_ft} Feet", "FULL", "ROUND", "NONE",
            None,
            "PLANAR",
        )

        route_buffer_dissolved_fc = os.path.join("in_memory", "route_buf_d")
        if arcpy.Exists(route_buffer_dissolved_fc):
            arcpy.management.Delete(route_buffer_dissolved_fc)
        arcpy.management.Dissolve(route_buffer_fc, route_buffer_dissolved_fc)

        arcpy.management.MakeFeatureLayer(parcels_roi_fc, "parcels_roi_fl")
        arcpy.management.SelectLayerByLocation(
            "parcels_roi_fl",
            "INTERSECT",
            projected_points_fc,
            selection_type="NEW_SELECTION",
        )

        parcels_with_points_fc = os.path.join("in_memory","parcels_has_pts")
        parcels_without_points_fc  = os.path.join("in_memory","parcels_no_pts")
        for fc_path in (parcels_with_points_fc, parcels_without_points_fc):
            if arcpy.Exists(fc_path): arcpy.management.Delete(fc_path)

        if int(arcpy.management.GetCount("parcels_roi_fl")[0]) > 0:
            arcpy.management.CopyFeatures("parcels_roi_fl", parcels_with_points_fc)
            arcpy.management.SelectLayerByAttribute("parcels_roi_fl", "SWITCH_SELECTION")

            if int(arcpy.management.GetCount("parcels_roi_fl")[0]) > 0:
                arcpy.management.CopyFeatures("parcels_roi_fl", parcels_without_points_fc)
            else:
                arcpy.management.CreateFeatureclass(
                    "in_memory",
                    "parcels_no_pts",
                    "POLYGON",
                    spatial_reference=working_spatial_ref,
                )
        else:
            arcpy.management.CopyFeatures(parcels_roi_fc, parcels_without_points_fc)
            arcpy.management.CreateFeatureclass(
                "in_memory",
                "parcels_has_pts",
                "POLYGON",
                spatial_reference=working_spatial_ref,
            )

        carved_parcels_fc = os.path.join("in_memory","carved_has_pts")
        if arcpy.Exists(carved_parcels_fc):
            arcpy.management.Delete(carved_parcels_fc)

        if int(arcpy.management.GetCount(parcels_with_points_fc)[0]) > 0:
            try:
                arcpy.analysis.PairwiseErase(
                    parcels_with_points_fc,
                    route_buffer_dissolved_fc,
                    carved_parcels_fc,
                )
            except Exception:
                arcpy.analysis.Erase(
                    parcels_with_points_fc,
                    route_buffer_dissolved_fc,
                    carved_parcels_fc,
                )
        else:
            arcpy.management.CreateFeatureclass(
                "in_memory",
                "carved_has_pts",
                "POLYGON",
                spatial_reference=working_spatial_ref,
            )

        merged_parcels_all_fc = os.path.join("in_memory","parcels_merged_all")
        if arcpy.Exists(merged_parcels_all_fc):
            arcpy.management.Delete(merged_parcels_all_fc)
        arcpy.management.CopyFeatures(parcels_without_points_fc, merged_parcels_all_fc)
        arcpy.management.Append(carved_parcels_fc, merged_parcels_all_fc, schema_type="NO_TEST")

        merged_edges_fc = os.path.join("in_memory","merged_edges")
        if arcpy.Exists(merged_edges_fc):
            arcpy.management.Delete(merged_edges_fc)
        arcpy.management.PolygonToLine(merged_parcels_all_fc, merged_edges_fc, "IDENTIFY_NEIGHBORS")

        # -------------------------------------------------
        # Build d2 rays and slit lines
        # -------------------------------------------------
        d2_rays_fc = os.path.join("in_memory","d2_rays")
        if arcpy.Exists(d2_rays_fc): arcpy.management.Delete(d2_rays_fc)

        arcpy.management.CreateFeatureclass(
            "in_memory",
            "d2_rays",
            "POLYLINE",
            spatial_reference=working_spatial_ref,
        )
        arcpy.management.AddField(d2_rays_fc,"IDX","LONG")
        arcpy.management.AddField(d2_rays_fc,"Bearing","DOUBLE")
        arcpy.management.AddField(d2_rays_fc,"Side","TEXT",field_length=1)

        side_label = "R" if camera_side.lower().startswith("r") else "L"

        point_index_to_ray = {}
        with arcpy.da.InsertCursor(d2_rays_fc, ["SHAPE@","IDX","Bearing","Side"]) as insert_cursor:
            for point_index, source_point in enumerate(source_points):
                bearing = float(bearings[point_index])
                side_offset_angle = 90.0 if camera_side.lower().startswith("r") else -90.0
                angle = math.radians((bearing + side_offset_angle) % 360.0)
                dx = math.sin(angle)
                dy = math.cos(angle)

                ray_end = arcpy.Point(
                    source_point.X + dx * ray_length_units,
                    source_point.Y + dy * ray_length_units,
                )
                ray_geometry = arcpy.Polyline(arcpy.Array([source_point, ray_end]), working_spatial_ref)

                insert_cursor.insertRow([ray_geometry, point_index, bearing, side_label])
                point_index_to_ray[point_index] = (ray_geometry, bearing)

        crossing_fc = os.path.join("in_memory","d2_crossings")
        if arcpy.Exists(crossing_fc):
            arcpy.management.Delete(crossing_fc)

        try:
            arcpy.analysis.PairwiseIntersect(
                [d2_rays_fc, merged_edges_fc],
                crossing_fc,
                "ALL",
                output_type="POINT",
            )
        except Exception:
            arcpy.analysis.Intersect(
                [d2_rays_fc, merged_edges_fc],
                crossing_fc,
                "ALL",
                output_type="POINT")

        point_index_to_measures = defaultdict(list)
        with arcpy.da.SearchCursor(crossing_fc, ["SHAPE@","IDX"]) as search_cursor:
            for shape, point_index in search_cursor:
                if point_index is None: continue

                ray_geometry, _ = point_index_to_ray.get(int(point_index), (None, None))
                if ray_geometry is None: continue

                if not isinstance(shape, arcpy.PointGeometry):
                    try:
                        first_point = getattr(shape, "firstPoint", None) or shape.getPart(0)[0]
                        shape = arcpy.PointGeometry(arcpy.Point(first_point.X, first_point.Y), working_spatial_ref)
                    except Exception:
                        continue

                measure = ray_geometry.measureOnLine(shape, use_percentage=False)
                if measure is not None and measure >= 0:
                    point_index_to_measures[int(point_index)].append(float(measure))

        slit_lines_fc = os.path.join("in_memory","slit_lines")
        if arcpy.Exists(slit_lines_fc): arcpy.management.Delete(slit_lines_fc)

        arcpy.management.CreateFeatureclass(
            "in_memory",
            "slit_lines",
            "POLYLINE",
            spatial_reference=working_spatial_ref,
        )

        slit_insert_cursor = arcpy.da.InsertCursor(slit_lines_fc, ["SHAPE@"])
        for point_index, (ray_geometry, bearing) in point_index_to_ray.items():
            measures = sorted(set(point_index_to_measures.get(point_index, [])))
            if len(measures) >= 2:
                first_measure, second_measure = measures[0], measures[1]
                d2_gap = second_measure - first_measure

                if 0.0 < d2_gap < d2_threshold_units:
                    segment_start = max(0.0, first_measure - d2_offset_units)
                    segment_end   = min(ray_geometry.length, second_measure + d2_offset_units)

                    try:
                        segment = ray_geometry.segmentAlongLine(segment_start, segment_end, use_percentage=False)
                        if segment and segment.length > 0:
                            slit_insert_cursor.insertRow([segment])
                    except Exception:
                        pass

        try: del slit_insert_cursor
        except Exception: pass

        if int(arcpy.management.GetCount(slit_lines_fc)[0]) > 0:
            slit_buffer_fc  = os.path.join("in_memory","slit_buf")
            slit_buffer_dissolved_fc = os.path.join("in_memory","slit_buf_d")

            for fc_path in (slit_buffer_fc, slit_buffer_dissolved_fc):
                if arcpy.Exists(fc_path):
                    arcpy.management.Delete(fc_path)

            arcpy.analysis.Buffer(
                slit_lines_fc,
                slit_buffer_fc,
                f"{carve_half_ft} Feet",
                "FULL",
                "ROUND",
                "ALL",
                None,
                "PLANAR",
            )
            arcpy.management.Dissolve(slit_buffer_fc, slit_buffer_dissolved_fc)

            merged_after_slit_fc = os.path.join("in_memory","merged_after_slit")
            if arcpy.Exists(merged_after_slit_fc):
                arcpy.management.Delete(merged_after_slit_fc)

            try:
                arcpy.analysis.PairwiseErase(
                    merged_parcels_all_fc,
                    slit_buffer_dissolved_fc,
                    merged_after_slit_fc,
                )
            except Exception:
                arcpy.analysis.Erase(
                    merged_parcels_all_fc,
                    slit_buffer_dissolved_fc,
                    merged_after_slit_fc,
                )
            parcels_for_use_fc = merged_after_slit_fc
        else:
            parcels_for_use_fc = merged_parcels_all_fc

        arcpy.AddMessage(f"[Parcels ready] using {int(arcpy.management.GetCount(parcels_for_use_fc)[0])} feature(s)")

        # -------------------------------------------------
        # Preprocess house polygons
        # -------------------------------------------------
        houses_available = False
        house_layer_name = None

        if house_path:
            try:
                if int(arcpy.management.GetCount(house_path)[0]) > 0:
                    arcpy.management.MakeFeatureLayer(house_path, "houses_fl_all")
                    arcpy.management.SelectLayerByLocation(
                        "houses_fl_all",
                        "INTERSECT",
                        roi_fc,
                        selection_type="NEW_SELECTION",
                    )
                    house_count = int(arcpy.management.GetCount("houses_fl_all")[0])

                    if house_count > 0:
                        houses_roi_fc = os.path.join("in_memory","houses_roi")
                        if arcpy.Exists(houses_roi_fc):
                            arcpy.management.Delete(houses_roi_fc)

                        arcpy.management.CopyFeatures("houses_fl_all", houses_roi_fc)
                        house_layer_name = "houses_fl"

                        try:
                            if arcpy.Exists(house_layer_name): arcpy.management.Delete(house_layer_name)
                        except Exception: pass

                        arcpy.management.MakeFeatureLayer(houses_roi_fc, house_layer_name)
                        houses_available = True
            except Exception:
                houses_available = False

        if not houses_available:
            arcpy.AddWarning("[HOUSES] No houses are found. FanHit=-1 and IsSurrounding=0.")

        # -------------------------------------------------
        # Transform points
        # -------------------------------------------------
        short_rays_fc = os.path.join("in_memory","rays_short")
        if arcpy.Exists(short_rays_fc):
            arcpy.management.Delete(short_rays_fc)

        arcpy.management.CreateFeatureclass(
            "in_memory",
            "rays_short",
            "POLYLINE",
            spatial_reference=working_spatial_ref
        )
        arcpy.management.AddField(short_rays_fc,"RID_","LONG")

        parcel_oid_name = arcpy.Describe(parcels_for_use_fc).OIDFieldName
        direction_char = "R" if camera_side.lower().startswith("r") else "L"

        def _fan_hit(source_point, bearing):
            if not houses_available or not house_layer_name:
                return -1

            side_offset_angle = 90.0 if camera_side.lower().startswith("r") else -90.0
            base_angle = (float(bearing) + side_offset_angle) % 360.0

            arc_step_count = 8
            fan_arc_points = []
            for angle_offset in np.linspace(-fan_half_deg, +fan_half_deg, arc_step_count):
                angle_rad = math.radians((base_angle + angle_offset) % 360.0)
                dx = math.sin(angle_rad)
                dy = math.cos(angle_rad)
                fan_arc_points.append(
                    arcpy.Point(
                        source_point.X + dx * fan_radius_units,
                        source_point.Y + dy * fan_radius_units,
                    )
                )

            fan_polygon = arcpy.Polygon(
                arcpy.Array(
                    [arcpy.Point(source_point.X, source_point.Y)]
                     + fan_arc_points
                     + [arcpy.Point(source_point.X, source_point.Y)]
                ),
                working_spatial_ref,
            )

            try:
                arcpy.management.SelectLayerByLocation(
                    house_layer_name,
                    "INTERSECT",
                    fan_polygon,
                    selection_type="NEW_SELECTION"
                )
                hit = 1 if int(arcpy.management.GetCount(house_layer_name)[0]) > 0 else 0
            finally:
                try:
                    arcpy.management.SelectLayerByAttribute(house_layer_name, "CLEAR_SELECTION")
                except Exception:
                    pass

            return hit

        inserted_count = 0
        with arcpy.da.InsertCursor(
            output_feature_class,
            ["SHAPE@", "SeqID", "FrameID", "ImagePath", "ParcelAddress", "IsSurrounding"],
        ) as output_insert_cursor:
            total_points = len(source_points)
            batch_size = min(2000, max(600, total_points//10 or total_points))
            batch_start = 0

            while batch_start < total_points:
                batch_end = min(batch_start + batch_size, total_points)

                arcpy.management.DeleteRows(short_rays_fc)
                with arcpy.da.InsertCursor(short_rays_fc, ["SHAPE@","RID_"]) as short_rays_cursor:
                    for ray_id, point_index in enumerate(range(batch_start, batch_end)):
                        source_point = source_points[point_index]
                        side_offset_angle = 90.0 if camera_side.lower().startswith("r") else -90.0
                        angle_rad = math.radians((float(bearings[point_index]) + side_offset_angle) % 360.0)
                        dx = math.sin(angle_rad)
                        dy = math.cos(angle_rad)

                        short_end = arcpy.Point(
                            source_point.X + dx * hit_window_units,
                            source_point.Y + dy * hit_window_units,
                        )

                        short_ray_geometry = arcpy.Polyline(
                            arcpy.Array([source_point, short_end]),
                            working_spatial_ref,
                        )

                        short_rays_cursor.insertRow([short_ray_geometry, ray_id])

                spatial_join_fc = os.path.join("in_memory", "sj_rays_parcels")
                if arcpy.Exists(spatial_join_fc):
                    arcpy.management.Delete(spatial_join_fc)

                field_mappings = arcpy.FieldMappings()
                field_mappings.addTable(short_rays_fc)

                parcel_oid_field_map = arcpy.FieldMap()
                parcel_oid_field_map.addInputField(parcels_for_use_fc, parcel_oid_name)
                output_field = parcel_oid_field_map.outputField
                output_field.name = "PARCEL_OID"
                output_field.aliasName = "PARCEL_OID"
                parcel_oid_field_map.outputField = output_field
                field_mappings.addFieldMap(parcel_oid_field_map)

                arcpy.analysis.SpatialJoin(
                    target_features=short_rays_fc,
                    join_features=parcels_for_use_fc,
                    out_feature_class=spatial_join_fc,
                    join_operation="JOIN_ONE_TO_MANY",
                    join_type="KEEP_COMMON",
                    field_mapping=field_mappings,
                    match_option="INTERSECT"
                )

                ray_id_to_parcel_oids = defaultdict(set)
                if int(arcpy.management.GetCount(spatial_join_fc)[0]) > 0:
                    with arcpy.da.SearchCursor(spatial_join_fc, ["RID_","PARCEL_OID"]) as search_cursor:
                        for ray_id, parcel_oid in search_cursor:
                            if parcel_oid is not None:
                                ray_id_to_parcel_oids[int(ray_id)].add(int(parcel_oid))

                arcpy.management.Delete(spatial_join_fc)

                needed_parcel_oids = set()
                for oid_set in ray_id_to_parcel_oids.values():
                    needed_parcel_oids.update(oid_set)

                parcel_oid_to_shape = {}
                if needed_parcel_oids:
                    with arcpy.da.SearchCursor(parcels_for_use_fc, [parcel_oid_name, "SHAPE@"]) as search_cursor:
                        for parcel_oid, parcel_shape in search_cursor:
                            if int(parcel_oid) in needed_parcel_oids:
                                parcel_oid_to_shape[int(parcel_oid)] = parcel_shape

                ray_id_to_parcel_shapes = defaultdict(list)
                for ray_id, oid_set in ray_id_to_parcel_oids.items():
                    for parcel_oid in oid_set:
                        parcel_shape = parcel_oid_to_shape.get(int(parcel_oid))
                        if parcel_shape:
                            ray_id_to_parcel_shapes[ray_id].append(parcel_shape)

                for ray_id, point_index in enumerate(range(batch_start,batch_end)):
                    source_point = source_points[point_index]
                    bearing = float(bearings[point_index])

                    side_offset_angle = 90.0 if camera_side.lower().startswith("r") else -90.0
                    angle_rad = math.radians((bearing + side_offset_angle) % 360.0)
                    dx = math.sin(angle_rad)
                    dy = math.cos(angle_rad)

                    fan_hit = _fan_hit(source_point, bearing)
                    is_surrounding  = 1 if (fan_hit == 0 and houses_available) else 0

                    short_end = arcpy.Point(
                        source_point.X + dx * hit_window_units,
                        source_point.Y + dy * hit_window_units,
                    )

                    short_ray_geometry = arcpy.Polyline(
                        arcpy.Array([source_point, short_end]),
                        working_spatial_ref
                    )

                    best_distance = None
                    best_parcel_shape = None
                    best_entry_point = None

                    for parcel_shape in ray_id_to_parcel_shapes.get(ray_id, []):
                        segment = parcel_shape.intersect(short_ray_geometry, 2)
                        if segment and segment.pointCount > 0:
                            for part_index in range(segment.partCount):
                                part = segment.getPart(part_index)
                                if not part or getattr(part,"count",0) < 2:
                                    continue

                                point_a = arcpy.PointGeometry(part[0], working_spatial_ref)
                                point_b = arcpy.PointGeometry(part[part.count-1], working_spatial_ref)

                                measure_a = short_ray_geometry.measureOnLine(point_a)
                                measure_b = short_ray_geometry.measureOnLine(point_b)

                                if measure_a < measure_b:
                                    candidate_distance = measure_a
                                    candidate_point = point_a
                                else:
                                    candidate_distance = measure_b
                                    candidate_point = point_b

                                if candidate_distance >= 0 and (
                                    best_distance is None or candidate_distance < best_distance
                                ):
                                    best_distance = candidate_distance
                                    best_parcel_shape = parcel_shape
                                    best_entry_point = candidate_point

                    moved = 0
                    if best_distance is None or best_distance > hit_window_units:
                        target_point = source_point
                    else:
                        entry_first_point = best_entry_point.firstPoint
                        extended_end = arcpy.Point(
                            entry_first_point.X + dx * long_probe_units,
                            entry_first_point.Y + dy * long_probe_units,
                        )
                        extended_ray = arcpy.Polyline(
                            arcpy.Array([entry_first_point, extended_end]),
                            working_spatial_ref,
                        )

                        inside_distance_units = None
                        inside_segment = best_parcel_shape.intersect(extended_ray, 2)

                        if inside_segment and inside_segment.pointCount > 0:
                            min_start_measure = None
                            for part_index in range(inside_segment.partCount):
                                part = inside_segment.getPart(part_index)
                                if not part or getattr(part,"count",0) < 2:
                                    continue

                                point_a = arcpy.PointGeometry(part[0], working_spatial_ref)
                                point_b = arcpy.PointGeometry(part[part.count-1], working_spatial_ref)

                                measure_a = extended_ray.measureOnLine(point_a)
                                measure_b = extended_ray.measureOnLine(point_b)
                                segment_start, segment_end = (measure_a, measure_b) if measure_a <= measure_b else (measure_b, measure_a)

                                if min_start_measure is None or segment_start < min_start_measure:
                                    min_start_measure = segment_start
                                    inside_distance_units = segment_end - segment_start

                        if inside_distance_units is None:
                            final_distance = best_distance + threshold_ft * feet_to_units
                        else:
                            inside_distance_ft = inside_distance_units * units_to_feet
                            extra_distance_ft = threshold_ft if inside_distance_ft > threshold_ft else 0.5 * inside_distance_ft
                            final_distance = min(
                                best_distance + extra_distance_ft * feet_to_units,
                                best_distance + inside_distance_units,
                            )

                        target_point = arcpy.Point(
                            source_point.X + dx * final_distance,
                            source_point.Y + dy * final_distance,
                        )
                        moved = 1

                    sequence_id = point_index
                    frame_id = frame_start_id + sequence_id
                    image_path = frame_id_to_path.get(frame_id, "")

                    output_insert_cursor.insertRow(
                        [
                            arcpy.PointGeometry(target_point, working_spatial_ref),
                            sequence_id,
                            frame_id,
                            image_path,
                            "",
                            is_surrounding,
                        ]
                    )
                    inserted_count += 1

                batch_start = batch_end

        arcpy.AddMessage(f"[Transform] {inserted_count} features")

        # -------------------------------------------------
        # Populate parcel address
        # -------------------------------------------------
        address_join_fc = os.path.join("in_memory", "sj_tp")
        if arcpy.Exists(address_join_fc):
            arcpy.management.Delete(address_join_fc)

        arcpy.analysis.SpatialJoin(
            target_features=output_feature_class,
            join_features=parcels_for_use_fc,
            out_feature_class=address_join_fc,
            join_operation="JOIN_ONE_TO_MANY",
            join_type="KEEP_COMMON",
            match_option="INTERSECT"
        )

        joined_field_names = [field.name for field in arcpy.ListFields(address_join_fc)]
        frame_id_field = (
            "FrameID" if "FrameID" in joined_field_names else
             next((f for f in joined_field_names if f.lower()=="frameid"), None)
        )
        if not frame_id_field:
            raise arcpy.ExecuteError("FrameID not found in join output.")

        def _pick_field(candidate_names):
            for candidate in candidate_names:
                if candidate in joined_field_names:
                    return candidate

            def normalize(text):
                return text.replace("_", "").replace(" ", "").lower()

            for field_name in joined_field_names:
                normalized_field = normalize(field_name)
                for candidate in candidate_names:
                    if normalized_field == normalize(candidate) or normalized_field.startswith(normalize(candidate)):
                        return field_name
            return None

        address_field  = _pick_field(
            ["Own_Addres","Own_Address","Address","SiteAddres","Situs_Addre","Situs_Address","Prop_Addr","PropAddress"]
        )
        legal_field = _pick_field(
            ["Legal Description","Legal_Description","LEGALDESC","LEGAL_DESCR","LegalDescr","LEGALDESCRIPTION"]
        )
        pin_field   = "PIN" if "PIN" in joined_field_names else _pick_field(["PIN"])

        if not (address_field or legal_field):
            arcpy.AddWarning("No parcel address field was found in the join output. 'Parcel Address' will remain blank.")
            frame_id_to_address = {}
        else:
            selected_fields = [frame_id_field]
            if address_field:
                selected_fields.append(address_field)
            if legal_field:
                selected_fields.append(legal_field)
            if pin_field:
                selected_fields.append(pin_field)

            rows = [tuple(row) for row in arcpy.da.SearchCursor(address_join_fc, selected_fields)]

            df_columns = ["FrameID"]
            if address_field:
                df_columns.append("Addr")
            if legal_field:
                df_columns.append("Legal")
            if pin_field:
                df_columns.append("PIN")

            join_df = pd.DataFrame(rows, columns=df_columns) if rows else pd.DataFrame(columns=df_columns)

            if not join_df.empty:
                join_df["FrameID"] = pd.to_numeric(join_df["FrameID"], errors="coerce").astype("Int64")
                join_df = join_df.dropna(subset=["FrameID"]).copy()
                join_df["FrameID"] = join_df["FrameID"].astype(int)

                def _first_nonempty(value_a, value_b):
                    value_a = ("" if value_a is None else str(value_a)).strip()
                    value_b = ("" if value_b is None else str(value_b)).strip()
                    return value_a if value_a else value_b

                if "Addr" in join_df.columns and "Legal" in join_df.columns:
                    join_df["Label"] = [
                        _first_nonempty(addr_value, legal_value)
                        for addr_value, legal_value in zip(join_df["Addr"], join_df["Legal"])]
                elif "Addr" in join_df.columns:
                    join_df["Label"] = join_df["Addr"].astype("string").fillna("")
                else:
                    join_df["Label"] = join_df["Legal"].astype("string").fillna("")

                if "PIN" in join_df.columns:
                    join_df["JoinedAddress"] = (
                        join_df["Label"].astype("string")
                       .str.cat(join_df["PIN"].astype("string"), sep=" ", na_rep="")
                       .str.strip()
                    )
                else:
                    join_df["JoinedAddress"] = join_df["Label"].astype("string").str.strip()

                frame_id_to_address = (
                    join_df.dropna(subset=["JoinedAddress"])
                    .drop_duplicates("FrameID")
                    .set_index("FrameID")["JoinedAddress"]
                    .to_dict()
                )
            else:
                frame_id_to_address = {}

        if frame_id_to_address:
            temp_address_table = arcpy.CreateUniqueName("addr_tbl", "in_memory")
            arcpy.management.CreateTable("in_memory", Path(temp_address_table).name)
            arcpy.management.AddField(temp_address_table, "FrameID", "LONG")
            arcpy.management.AddField(temp_address_table, "ParcelAddrTmp", "TEXT", field_length=255)

            with arcpy.da.InsertCursor(temp_address_table, ["FrameID", "ParcelAddrTmp"]) as insert_cursor:
                for frame_id, address_text in frame_id_to_address.items():
                    try:
                        insert_cursor.insertRow((int(frame_id), "" if address_text is None else str(address_text)[:255]))
                    except Exception:
                        continue

            existing_output_fields = {field.name for field in arcpy.ListFields(output_feature_class)}
            if "ParcelAddrTmp" in existing_output_fields:
                arcpy.management.DeleteField(output_feature_class, ["ParcelAddrTmp"])

            arcpy.management.JoinField(
                output_feature_class,
                "FrameID",
                temp_address_table,
                "FrameID",
                ["ParcelAddrTmp"],
            )
            arcpy.management.CalculateField(
                output_feature_class,
                "ParcelAddress",
                "!ParcelAddrTmp!",
                "PYTHON3"
            )
            arcpy.management.DeleteField(output_feature_class, ["ParcelAddrTmp"])

        # -------------------------------------------------
        # Attachments
        # -------------------------------------------------
        def _safe_fc_name(name, gdb):
            try:
                return arcpy.ValidateTableName(os.path.basename(name), gdb)
            except Exception:
                return re.sub(r'[^A-Za-z0-9_]+', '_', os.path.basename(name))[:120]

        def _clear_locks():
            try:
                arcpy.ClearWorkspaceCache_management()
            except Exception:
                pass
            time.sleep(0.15)

        def _force_enable_attachments(fc_path):
            desc = arcpy.Describe(fc_path)
            gdb_path  = desc.path
            if not gdb_path or ".gdb" not in gdb_path.lower():
                raise arcpy.ExecuteError("Attachments require a file geodatabase output. Set 'Output Points' to a .gdb feature class.")

            old_workspace = arcpy.env.workspace
            arcpy.env.workspace = gdb_path

            safe_name = _safe_fc_name(fc_path, gdb_path)
            safe_fc_path   = os.path.join(gdb_path, safe_name)

            if os.path.normpath(safe_fc_path) != os.path.normpath(fc_path):
                arcpy.AddMessage(f"Renaming output to safe name '{safe_name}' for attachments.")
                arcpy.management.Rename(fc_path, safe_name, "FeatureClass")
                fc_path = safe_fc_path

            attachment_table = os.path.join(gdb_path, f"{safe_name}__ATTACH")
            attachment_rel = os.path.join(gdb_path, f"{safe_name}__ATTACHREL")

            for object_path in (attachment_rel, attachment_table):
                if arcpy.Exists(object_path):
                    try:
                        arcpy.AddMessage(f"Deleting orphan: {object_path}")
                        arcpy.management.Delete(object_path)
                    except Exception as e:
                        arcpy.AddWarning(f"Could not delete orphan '{object_path}': {e}")

            if not getattr(arcpy.Describe(fc_path), "hasGlobalID", False):
                arcpy.AddMessage("Adding GlobalIDs to output…")
                arcpy.management.AddGlobalIDs(fc_path)

            try:
                if getattr(arcpy.Describe(fc_path), "hasAttachments", False):
                    arcpy.management.DisableAttachments(fc_path)
            except Exception:
                pass

            _clear_locks()
            try:
                arcpy.management.EnableAttachments(fc_path)
            except Exception as e:
                _clear_locks()
                for object_path in (attachment_rel, attachment_table):
                    if arcpy.Exists(object_path):
                        try: arcpy.management.Delete(object_path)
                        except Exception: pass
                arcpy.management.EnableAttachments(fc_path)

            attachment_table_exists = arcpy.Exists(attachment_table)
            attachment_rel_exists = arcpy.Exists(attachment_rel)

            if not (attachment_table_exists and attachment_rel_exists):
                time.sleep(0.25)
                _clear_locks()
                attachment_table_exists = arcpy.Exists(attachment_table)
                attachment_rel_exists = arcpy.Exists(attachment_rel)

            arcpy.env.workspace = old_workspace

            if not (attachment_table_exists and attachment_rel_exists):
                raise arcpy.ExecuteError(
                    "EnableAttachments completed but the attachment table was not created. "
                    "Check FGDB write permissions and close any schema locks on the output feature class."
                )

            return fc_path

        if attach_frames:
            rows_for_attachment = []
            with arcpy.da.SearchCursor(output_feature_class, ["FrameID","ImagePath"]) as search_cursor:
                for frame_id, image_path in search_cursor:
                    if image_path and os.path.isfile(image_path):
                        rows_for_attachment.append((int(frame_id), image_path))

            if not rows_for_attachment:
                raise arcpy.ExecuteError("No image files are found to attach.")

            output_feature_class = _force_enable_attachments(output_feature_class)

            temp_attachment_table = arcpy.CreateUniqueName("att_tbl", "in_memory")
            arcpy.management.CreateTable("in_memory", Path(temp_attachment_table).name)
            arcpy.management.AddField(temp_attachment_table, "FrameID", "LONG")
            arcpy.management.AddField(temp_attachment_table, "ATTACHMENT", "TEXT", field_length=500)

            with arcpy.da.InsertCursor(temp_attachment_table, ["FrameID","ATTACHMENT"]) as insert_cursor:
                for row in rows_for_attachment:
                    insert_cursor.insertRow(row)

            arcpy.management.AddAttachments(
                output_feature_class,
                "FrameID",
                temp_attachment_table,
                "FrameID",
                "ATTACHMENT",
            )
            parameters[9].value = output_feature_class
            arcpy.AddMessage("Attachments added successfully.")
        else:
            parameters[9].value = output_feature_class
            arcpy.AddMessage("Attach Frames unchecked; skipping attachment step.")

        # -------------------------------------------------
        # Output address folders
        # -------------------------------------------------
        if address_folders and frames_available:
            frame_id_to_is_surrounding = {}
            with arcpy.da.SearchCursor(output_feature_class, ["FrameID", "IsSurrounding"]) as search_cursor:
                for frame_id, is_surrounding in search_cursor:
                    try:
                        frame_id_to_is_surrounding[int(frame_id)] = int(is_surrounding or 0)
                    except Exception:
                        continue

            base_folder = Path(address_folders)
            base_folder.mkdir(parents=True, exist_ok=True)

            if refresh_folders:
                for subfolder in list(base_folder.iterdir()):
                    if subfolder.is_dir():
                        shutil.rmtree(subfolder, ignore_errors=True)

            frame_files = []
            with os.scandir(frames_folder) as scan_iter:
                for entry in scan_iter:
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in extensions:
                        match_id = frame_number_pattern.search(entry.name)
                        if match_id:
                            frame_files.append((int(match_id.group(1)), entry.path))

            def safe_folder_name(text):
               return re.sub(r'[<>:"/\|?*]+', "_", str(text))[:150]

            copied_count = 0
            created_address_folders = set()

            for frame_id, file_path in frame_files:
                address_text = frame_id_to_address.get(frame_id)
                if address_text and address_text.strip():
                    address_folder = base_folder / safe_folder_name(address_text)

                    if address_folder not in created_address_folders:
                        address_folder.mkdir(parents=True, exist_ok=True)
                        created_address_folders.add(address_folder)
                        (address_folder / "Surrounding").mkdir(parents=True, exist_ok=True)

                    is_surrounding = 1 if frame_id_to_is_surrounding.get(frame_id, 0) == 1 else 0
                    destination_folder = address_folder / ("Surrounding" if is_surrounding == 1 else "")

                    try:
                        shutil.copy2(file_path, destination_folder if destination_folder != address_folder else address_folder)
                        copied_count += 1
                    except Exception as e:
                        arcpy.AddWarning(f"Copy failed {file_path} → {destination_folder}: {e}")

            arcpy.AddMessage(
                        f"Frames matched to addresses: {copied_count}."
                        f"Created {len(created_address_folders)} address folder(s) with 'Surrounding' subfolder."
                    )

        elif address_folders and not frames_available:
            arcpy.AddWarning(
                f"Output Addresses Folder provided, but Frames Folder is not provided."
                f"Skipping creating address folders."
            )
        else:
            arcpy.AddMessage("Output Addresses Folder not provided; skipping creating address folders.")

        # -------------------------------------------------
        # Keep only final user-facing fields
        # -------------------------------------------------
        existing_fields = {field.name for field in arcpy.ListFields(output_feature_class)}
        field_to_drop = [field_name for field_name in ["SeqID", "FrameID"] if field_name in existing_fields]
        if field_to_drop:
            arcpy.management.DeleteField(output_feature_class, field_to_drop)

        parameters[9].value = output_feature_class
        arcpy.AddMessage(f"[Total runtime] {time.time() - start_time:.2f}s")