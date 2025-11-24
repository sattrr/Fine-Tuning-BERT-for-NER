def clean_conll(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f, \
         open(output_path, "w", encoding="utf-8") as w:
        
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                w.write("\n")
                continue
            
            parts = line.split()
            token = parts[0]
            label = parts[-1]

            if label.startswith("I-") and prev_label != label.replace("I-", "B-"):
                label = "B-" + label.split("-")[1]
            
            w.write(f"{token} {label}\n")
            prev_label = label

clean_conll("../data/gizi-klinis.conll",
            "../data/gizi-klinis.clean.conll")
