def merge_even_lines(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    merged_lines = []
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):  # sprawdzenie, czy istnieje następna linia
            merged_lines.append(lines[i].strip() + lines[i + 1])  # dodanie zawartości parzystej linii na koniec poprzedniej

    with open(output_file, 'w') as file:
        file.write(''.join(merged_lines))

# Użycie: podaj nazwę pliku wejściowego i pliku wyjściowego
input_file = 'input.txt'  # Zmień na nazwę swojego pliku wejściowego
output_file = 'output.txt'  # Zmień na nazwę pliku, do którego chcesz zapisać wynik
merge_even_lines(input_file, output_file)
