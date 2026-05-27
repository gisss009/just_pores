for i in range(1, 1502):
    if i == 901:
        continue

    input_path = rf"C:\Users\userd\Desktop\Новая папка\just_pores\dataset_pore\sample{i}\output_matrix.txt"
    output_path = rf"C:\Users\userd\Desktop\Новая папка\just_pores\update_dataset_pore\sample{i}.txt"

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.replace(" ", ""))