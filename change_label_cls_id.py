import os


def load_and_change_yolo_labels(directory):
    if not os.path.exists(directory):
        return
    replace_map = {
        2.0: "3",
        3.0: "4",
        4.0: "5",
        5.0: "6",
        6.0: "7",
        7.0: "9",
        8.0: "10",
        9.0: "11",
    }

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as f:
                modified_lines = []
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    key = float(parts[0])
                    if key in replace_map:
                        parts[0] = replace_map[key]

                    modified_line = " ".join(parts) + "\n"
                    modified_lines.append(modified_line)

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(modified_lines)


load_and_change_yolo_labels("LSB_receptaclev6_fine_tune_yolov11/valid/labels")
