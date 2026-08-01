import os
import re
import csv
from collections import defaultdict

import arcpy


# --------------------------------------------------------
# Helper functions
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

def find_field_case_insensitive(feature_class, wanted_name):
    wanted_key = re.sub(r"[\s_]+", "", wanted_name).lower()

    for field in arcpy.ListFields(feature_class) or []:
        field_key = re.sub(r"[\s_]+", "", field.name).lower()
        if field_key == wanted_key:
            return field.name

    return None

def parcel_key_to_text(parcel_key):
    if parcel_key is None:
        return None
    return f"{parcel_key[0]}|{parcel_key[1]}"

def get_matching_fields(feature_class):
    pin_field = find_field_case_insensitive(feature_class, "PIN")
    address_field = (
        find_field_case_insensitive(feature_class, "Own_Addres")
        or find_field_case_insensitive(feature_class, "Address")
    )
    recovery_score_field = find_field_case_insensitive(feature_class, "recovery_score")

    if recovery_score_field is None:
        raise arcpy.ExecuteError(
            f"Could not find 'recovery_score' field in: {feature_class}"
        )

    return pin_field, address_field, recovery_score_field

def make_parcel_key(pin_value, address_value):
    pin_key = extract_pin_from_text(pin_value)
    if pin_key:
        return ("PIN", pin_key)

    normalized_address = normalize_text(address_value)
    if normalized_address:
        return ("ADDR", normalized_address)

    return None

def make_csv_output_paths(csv_output_folder):
    percent_csv = os.path.join(csv_output_folder, "percent_recovered.csv")
    years_to_recover_csv = os.path.join(csv_output_folder, "years_to_recover.csv")
    return percent_csv, years_to_recover_csv

# --------------------------------------------------------
# Toolbox
# --------------------------------------------------------

class Toolbox(object):
    def __init__(self):
        self.label = "Perform Longitudinal Recovery"
        self.alias = "PerformLongitudinalAnalysis"
        self.tools = [PerformLongitudinalAnalysis]


class PerformLongitudinalAnalysis(object):
    def __init__(self):
        self.label = "Perform Longitudinal Recovery"
        self.description = (
            "Perform longitudinal analysis of recovery using housing recovery feature classes."
        )

    def getParameterInfo(self):
        year_fc_table = arcpy.Parameter(
            displayName="Year and Housing Recovery Feature Class",
            name="year_fc_table",
            datatype="GPValueTable",
            parameterType="Required",
            direction="Input",
        )
        year_fc_table.columns = [
            ["GPLong", "Year"],
            ["GPFeatureLayer", "Housing Recovery Feature Class"],
        ]
        year_fc_table.filters[1].list = ["Polygon"]

        output_feature_class = arcpy.Parameter(
            displayName="Output Longitudinal Recovery Feature Class",
            name="output_feature_class",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output",
        )

        csv_output_folder = arcpy.Parameter(
            displayName="Output Longitudinal Results Files",
            name="csv_output_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input",
        )
        return [
            year_fc_table,
            csv_output_folder,
            output_feature_class,
        ]

    def updateMessages(self, params):
        table_values = params[0].values
        csv_folder = params[1].valueAsText
        output_fc  = params[2].valueAsText

        if table_values:
            for row in table_values:
                if len(row) != 2:
                    params[0].setErrorMessage("Each row must contain exactly: Year and Housing Recovery Feature Class.")
                    return

                try:
                    int(row[0])
                except Exception:
                    params[0].setErrorMessage("Year values must be integers.")
                    return

        if output_fc:
            parent_path = os.path.dirname(output_fc)

            if not parent_path.lower().endswith(".gdb"):
                params[2].setErrorMessage(
                    "Output feature class must be created directly inside a file geodatabase, not inside a feature dataset."
                )

        if csv_folder and not os.path.isdir(csv_folder):
            params[1].setErrorMessage("CSV Output Folder must be an existing folder.")

    def execute(self, params, messages):
        table_values = params[0].values
        csv_output_folder = params[1].valueAsText
        output_feature_class = params[2].valueAsText

        if not table_values:
            raise arcpy.ExecuteError("No year or feature classes provided.")

        paired_inputs = []
        for row in table_values:
            year = int(row[0])
            feature_class = str(row[1])
            paired_inputs.append((year, feature_class))

        paired_inputs.sort(key=lambda x: x[0])

        percent_csv, years_to_recover_csv = make_csv_output_paths(csv_output_folder)

        # --------------------------------------------------------
        # Storage
        # --------------------------------------------------------
        parcel_records = {}
        parcel_pin = {}
        parcel_address = {}

        yearly_percent_rows = []
        years_to_recover_rows = []

        # --------------------------------------------------------
        # Read feature classes
        # --------------------------------------------------------
        arcpy.SetProgressor("step", "Reading layers...", 0, len(paired_inputs), 1)

        for i, (year, feature_class) in enumerate(paired_inputs, 1):
            arcpy.SetProgressorLabel(f"Reading year {year} ({i}/{len(paired_inputs)})")
            arcpy.SetProgressorPosition(i)

            pin_field, address_field, recovery_score_field = get_matching_fields(feature_class)

            cursor_fields = [recovery_score_field]
            if pin_field:
                cursor_fields.append(pin_field)
            if address_field:
                cursor_fields.append(address_field)

            count_recovered = 0
            count_1234 = 0

            with arcpy.da.SearchCursor(feature_class, cursor_fields) as cursor:
                for row in cursor:
                    idx = 0
                    recovery_score = row[idx]
                    idx += 1

                    pin_value = row[idx] if pin_field else None
                    if pin_field:
                        idx += 1

                    address_value = row[idx] if address_field else None

                    parcel_key = make_parcel_key(pin_value, address_value)
                    if parcel_key is None:
                        continue

                    # Count yearly recovered percentage
                    if recovery_score in (1, 2, 3, 4):
                        count_1234 += 1
                        if int(recovery_score) == 4:
                            count_recovered += 1

                    # Initialize parcel record
                    if parcel_key not in parcel_records:
                        parcel_records[parcel_key] = {
                            "score_history": [],
                            "observed_years": set(),
                            "first_year_seen": year,
                            "recovered_year": None,
                        }
                        parcel_pin[parcel_key] = pin_value
                        parcel_address[parcel_key] = address_value

                    record = parcel_records[parcel_key]
                    if recovery_score in (1, 2, 3, 4):
                        record["observed_years"].add(year)

                    # If parcel already recovered, stop storing additional scores
                    if record["recovered_year"] is not None:
                        continue

                    # Store valid scores only
                    if recovery_score in (1, 2, 3, 4):
                        record["score_history"].append(int(recovery_score))

                        if int(recovery_score) == 4:
                            record["recovered_year"] = year

            # Percentage of recovered buildings for that year
            if count_1234 > 0:
                percent_recovered = (count_recovered / float(count_1234)) * 100.0
            else:
                percent_recovered = 0.0

            yearly_percent_rows.append({
                "year": year,
                "percent_recovered_building": percent_recovered,
            })

        arcpy.ResetProgressor()

        # --------------------------------------------------------
        # Keep only parcels with full score histories
        # --------------------------------------------------------
        required_year_count = len(paired_inputs)

        filtered_parcel_records = {}
        for parcel_key, record in parcel_records.items():
            if len(record["observed_years"]) == required_year_count:
                filtered_parcel_records[parcel_key] = record

        parcel_records = filtered_parcel_records

        # --------------------------------------------------------
        # Build parcel-level years-to-recover rows
        # --------------------------------------------------------
        for parcel_key, record in parcel_records.items():
            if record["recovered_year"] is not None:
                years_to_recover_rows.append({
                    "PIN": parcel_pin.get(parcel_key),
                    "Own_Addres": parcel_address.get(parcel_key),
                    "RecoveredYr": record["recovered_year"],
                    "Years_To_Recover": len(record["score_history"]),
                })

        # --------------------------------------------------------
        # Create output parcel feature class
        # --------------------------------------------------------
        first_fc = paired_inputs[0][1]

        out_gdb = os.path.dirname(output_feature_class)
        final_name = os.path.basename(output_feature_class)
        temp_output_fc = os.path.join(out_gdb, f"{final_name}_tmp")
        results_table = os.path.join(out_gdb, f"{final_name}_results_tbl")

        if arcpy.Exists(temp_output_fc):
            arcpy.management.Delete(temp_output_fc)

        if arcpy.Exists(output_feature_class):
            arcpy.management.Delete(output_feature_class)

        if arcpy.Exists(results_table):
            arcpy.management.Delete(results_table)

        arcpy.conversion.FeatureClassToFeatureClass(
            first_fc,
            out_gdb,
            os.path.basename(temp_output_fc),
        )

        # --------------------------------------------------------
        # Delete unnecessary fields
        # --------------------------------------------------------
        fields_to_delete = []

        for field in arcpy.ListFields(temp_output_fc):
            field_name_lower = field.name.lower()

            if field.required:
                continue

            if field_name_lower in {
                "recovery_score",
                "recovery_state",
                "prediction_confidence",
                "globalid",
                "shape_length",
                "shape_area",
            }:
                fields_to_delete.append(field.name)

        if fields_to_delete:
            arcpy.management.DeleteField(temp_output_fc, fields_to_delete)

        # --------------------------------------------------------
        # Add results fields
        # --------------------------------------------------------
        existing_fields = {field.name.lower() for field in arcpy.ListFields(temp_output_fc)}

        if "parcelkey" not in existing_fields:
            arcpy.management.AddField(temp_output_fc, "ParcelKey", "TEXT", field_length=255)

        output_pin_field = find_field_case_insensitive(temp_output_fc, "PIN")
        output_address_field = (
            find_field_case_insensitive(temp_output_fc, "Own_Addres")
            or find_field_case_insensitive(temp_output_fc, "Address")
        )
        pin_expr = f"!{output_pin_field}!" if output_pin_field else "None"
        addr_expr = f"!{output_address_field}!" if output_address_field else "None"

        calc_code = r"""
def build_key(pin_value, address_value):
    import re

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

    pin_key = extract_pin_from_text(pin_value)
    if pin_key:
        return "PIN|" + pin_key

    normalized_address = normalize_text(address_value)
    if normalized_address:
        return "ADDR|" + normalized_address

    return None
"""

        arcpy.management.CalculateField(
            temp_output_fc,
            "ParcelKey",
            f"build_key({pin_expr}, {addr_expr})",
            "PYTHON3",
            calc_code,
        )

        # --------------------------------------------------------
        # Create results table
        # --------------------------------------------------------
        arcpy.management.CreateTable(out_gdb, os.path.basename(results_table))
        arcpy.management.AddField(results_table, "ParcelKey", "TEXT", field_length=255)
        arcpy.management.AddField(results_table, "FirstYear", "LONG")
        arcpy.management.AddField(results_table, "RecoveredYr", "LONG")
        arcpy.management.AddField(results_table, "Years_To_Re", "LONG")
        arcpy.management.AddField(results_table, "ScoreHist", "TEXT", field_length=255)

        with arcpy.da.InsertCursor(
            results_table,
            ["ParcelKey", "FirstYear", "RecoveredYr", "Years_To_Re", "ScoreHist"]
        ) as cursor:
            for parcel_key, record in parcel_records.items():
                score_history = record["score_history"]

                if len(record["observed_years"]) != required_year_count:
                    continue
                years_to_recover = len(score_history) if record["recovered_year"] is not None else None

                cursor.insertRow((
                    parcel_key_to_text(parcel_key),
                    record["first_year_seen"],
                    record["recovered_year"],
                    years_to_recover,
                    ",".join(str(score) for score in score_history),
                ))

        # --------------------------------------------------------
        # Join results onto temp feature class
        # --------------------------------------------------------
        arcpy.management.JoinField(
            temp_output_fc,
            "ParcelKey",
            results_table,
            "ParcelKey",
            ["FirstYear", "RecoveredYr", "Years_To_Re", "ScoreHist"],
        )

        # --------------------------------------------------------
        # Keep only parcels with full score histories
        # --------------------------------------------------------
        temp_layer = "longitudinal_tmp_layer"
        if arcpy.Exists(temp_layer):
            arcpy.management.Delete(temp_layer)

        arcpy.management.MakeFeatureLayer(temp_output_fc, temp_layer)
        arcpy.management.SelectLayerByAttribute(temp_layer, "NEW_SELECTION", "FirstYear IS NULL")
        arcpy.management.DeleteFeatures(temp_layer)

        try:
            arcpy.management.Delete(temp_layer)
        except Exception:
            pass
        # --------------------------------------------------------
        # Write percent recovered table
        # --------------------------------------------------------
        with open(percent_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["year", "% of recovered building"])

            for row in yearly_percent_rows:
                writer.writerow([
                    row["year"],
                    row["percent_recovered_building"],
                ])

        # --------------------------------------------------------
        # Write years-to-recover table
        # --------------------------------------------------------
        with open(years_to_recover_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["PIN", "Own_Addres", "RecoveredYr", "years to recover"])

            for row in years_to_recover_rows:
                writer.writerow([
                    row["PIN"],
                    row["Own_Addres"],
                    row["RecoveredYr"],
                    row["Years_To_Recover"],
                ])

        # --------------------------------------------------------
        # Finalize output
        # --------------------------------------------------------
        if arcpy.Exists(output_feature_class):
            arcpy.management.Delete(output_feature_class)

        arcpy.management.CopyFeatures(temp_output_fc, output_feature_class)

        try:
            arcpy.management.Delete(temp_output_fc)
        except Exception:
            pass

        try:
            arcpy.management.Delete(results_table)
        except Exception:
            pass

        params[2].value = output_feature_class

        arcpy.AddMessage(f"Output feature class: {output_feature_class}")
        arcpy.AddMessage(f"% recovered csv: {percent_csv}")
        arcpy.AddMessage(f"Years to recover csv: {years_to_recover_csv}")
        arcpy.AddMessage("Longitudinal analysis complete.")
