import sys
from pathlib import Path
import pandas as pd

import openpyxl
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox, QLabel, QLineEdit, QPushButton, \
    QFileDialog, QCheckBox, QHBoxLayout, QButtonGroup

from GeneratingTraces_MathematicalModel import RandomWalk
from GeneratingTraces_RGANtorch.FEM.generate_dataset import GenerateDataset

def get_combination_from_excel(excel_file):
    wb = openpyxl.load_workbook(excel_file)
    sheet = wb.active
    headers = [cell.value for cell in sheet[1]][0:4]
    all_combinations = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        tupel1 = dict(zip(headers, row[0:4]))
        all_combinations.append(tupel1)
    wb.close()
    return all_combinations
def transform_dict(input_dict):
    key_values = list(input_dict.items())[0]
    keys = key_values[0].split(';')
    values = map(float, key_values[1].split(';'))
    result_dict = dict(zip(keys, values))
    return result_dict

def get_combination_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    all_combinations = df.to_dict(orient='records')
    transformed_combinations = []
    for combination in all_combinations:
        transformed_combinations.append(transform_dict(combination))

    return transformed_combinations

class Functs:
    @staticmethod
    def getWalk():
        print("getWalk() wurde aufgerufen.")


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Hauptfenster-Einstellungen
        self.setWindowTitle("Create synthetic FEMs")
        self.setGeometry(200, 200, 400, 400)

        # Haupt-Widget und Layout erstellen
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Funktionsauswahl-ComboBox erstellen
        self.function_combo = QComboBox(self)
        self.label_for_func = QLabel("Method:")
        self.layout.addWidget(self.label_for_func)
        self.function_combo.addItem("RandomWalk")
        self.function_combo.addItem("RCGAN")
        self.layout.addWidget(self.function_combo)

        # Dropdown-Menü mit 16 Optionen erstellen
        self.float_options = []
        #excel_file = r"C:\Users\fanzl\bwSyncShare\Documents\Versuchsplanung Mathematisches Modell\AuswertungErgebnisse\ParameterInput.xlsx"
        # Open Excelfile
        combinations = get_combination_from_csv(Path("ParameterInputGui.csv"))
        for comb in combinations:
            self.float_options.append((comb['simulation rate'], comb['L'], comb['relaxation rate'], comb['h_crit']))

        self.drop_var = ['simulation_freq', 'grid size L', 'relaxation_rate', 'hc']
        self.float_combo = QComboBox(self)
        self.label_for_combo = QLabel("Parameter combination:")
        self.layout.addWidget(self.label_for_combo)
        for index, option in enumerate(self.float_options):
            self.float_combo.addItem(
                f"{self.drop_var[0]}={option[0]},{self.drop_var[1]}={option[1]},{self.drop_var[2]}={option[2]},{self.drop_var[3]}={option[3]}")
        self.layout.addWidget(self.float_combo)

        self.float_combo.currentIndexChanged.connect(self.toggle_input_fields_enabled)  # Event handler hinzufügen
        self.layout.addWidget(self.float_combo)

        self.file_label = QLabel("GAN Model File:")
        self.file_edit = QLineEdit()
        self.file_browse_button = QPushButton("Browse file")
        #self.file_browse_button.clicked.connect(self.browse_file("GAN Model Files (*.pth)"))
        self.label_file = QLabel("Label file:")
        self.label_file_edit = QLineEdit()
        self.label_file_browse_button = QPushButton("Browse file")
        #self.label_file_browse_button.clicked.connect(self.browse_file("Label files (*.csv)"))

        self.layout.addWidget(self.file_label)
        self.layout.addWidget(self.file_edit)
        self.layout.addWidget(self.file_browse_button)
        self.layout.addWidget(self.label_file)
        self.layout.addWidget(self.label_file_edit)
        self.layout.addWidget(self.label_file_browse_button)
        # Zuordnung von Anzeigenamen zu Variablennamen
        self.variable_mapping = {
                            "Number of simulations": "number",
                            "Duration": 'duration',
                            "Field size in degree": 'field_size',
                            "Sampling Frequency": 'sampling_frequency',
                            'Folderpath to save to': 'folderpath',
                            'Show plots': 'show_plots'
                            }
        # Felder für Variableneingabe erstellen
        self.input_fields = []
        self.default_checkboxes = []
        self.variable10_checkbox_group = QButtonGroup(self)  # ButtonGroup für Variable 10 Checkboxen
        self.variable10_checkbox_group.setExclusive(True)  # Nur eine Checkbox kann ausgewählt sein
        for display_name, variable_name in self.variable_mapping.items():
            if display_name == "Folderpath to save to":
                label = QLabel(variable_name + ":")
                line_edit = QLineEdit()
                browse_button = QPushButton("Browse folder")
                browse_button.clicked.connect(self.browse_folder)
                self.input_fields.append((variable_name, line_edit, browse_button))
                self.layout.addWidget(label)
                self.layout.addWidget(line_edit)
                self.layout.addWidget(browse_button)
            elif display_name == "Show plots":
                label = QLabel(variable_name + ":")
                self.checkbox_yes = QCheckBox("Yes")
                self.checkbox_no = QCheckBox("No")
                self.checkbox_yes.clicked.connect(lambda: self.handle_checkbox(self.checkbox_yes, self.checkbox_no))
                self.checkbox_no.clicked.connect(lambda: self.handle_checkbox(self.checkbox_no, self.checkbox_yes))
                self.input_fields.append((variable_name, self.checkbox_yes, self.checkbox_no))
                hbox = QHBoxLayout()
                hbox.addWidget(self.checkbox_yes)
                hbox.addWidget(self.checkbox_no)
                self.layout.addWidget(label)
                self.layout.addLayout(hbox)
            else:
                label = QLabel(display_name + ":")
                line_edit = QLineEdit()
                checkbox_default = QCheckBox("Default")
                checkbox_default.clicked.connect(
                    lambda state, line_edit=line_edit, checkbox_default=checkbox_default: self.toggle_input_field(
                        line_edit, checkbox_default))
                self.input_fields.append((variable_name, line_edit, checkbox_default))
                self.default_checkboxes.append(checkbox_default)
                hbox = QHBoxLayout()
                hbox.addWidget(line_edit)
                hbox.addWidget(checkbox_default)
                self.layout.addWidget(label)
                self.layout.addLayout(hbox)

        self.checkbox_no.setChecked(True)

        # Einheitenauswahl hinzufügen
        self.unit_label = QLabel("Unit:")
        self.unit_dva_checkbox = QCheckBox("DVA")
        self.unit_arcmin_checkbox = QCheckBox("Arcmin")
        self.unit_um_checkbox = QCheckBox("µm")

        self.unit_dva_checkbox.setChecked(True)

        # Hinzufügen der Checkboxen zur ButtonGroup, um sicherzustellen, dass nur eine gleichzeitig ausgewählt ist
        self.unit_checkbox_group = QButtonGroup(self)
        self.unit_checkbox_group.addButton(self.unit_dva_checkbox)
        self.unit_checkbox_group.addButton(self.unit_arcmin_checkbox)
        self.unit_checkbox_group.addButton(self.unit_um_checkbox)
        self.unit_checkbox_group.setExclusive(True)

        # Layout für die Einheiten-Checkboxen erstellen
        unit_layout = QHBoxLayout()
        unit_layout.addWidget(self.unit_dva_checkbox)
        unit_layout.addWidget(self.unit_arcmin_checkbox)
        unit_layout.addWidget(self.unit_um_checkbox)

        # Hinzufügen zur Haupt-Layout
        self.layout.addWidget(self.unit_label)
        self.layout.addLayout(unit_layout)

        # Button zum Ausführen der Funktion erstellen
        self.run_button = QPushButton("Start creating synthetic FEMs", self)
        self.run_button.clicked.connect(self.run_function)
        self.layout.addWidget(self.run_button)

        # Dateiauswahl-Widget erstellen

        self.function_combo.currentIndexChanged.connect(self.toggle_file_input_enabled)
        self.toggle_file_input_enabled()



    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Ordner auswählen")
        for element in self.input_fields:
            if len(element) == 3:
                variable_name, line_edit, _ = element
                if variable_name == "folderpath":
                    line_edit.setText(folder_path)

    def browse_file(self, fileformat):
        file_path, _ = QFileDialog.getOpenFileName(self, "Datei auswählen", "", fileformat)
        self.file_edit.setText(file_path)

    def toggle_input_field(self, line_edit, checkbox_default):
        line_edit.setDisabled(checkbox_default.isChecked())
    def toggle_file_input_enabled(self):
        selected_function = self.function_combo.currentText()
        enable_file_input = selected_function == "RCGAN"
        self.file_edit.setEnabled(enable_file_input)
        self.file_browse_button.setEnabled(enable_file_input)
        self.label_file_browse_button.setEnabled(enable_file_input)
        self.float_combo.setEnabled(not enable_file_input)
    def handle_checkbox(self, checkbox, other_checkbox):
        if checkbox.isChecked():
            other_checkbox.setChecked(False)

    def toggle_input_fields_enabled(self):
        selected_index = self.float_combo.currentIndex()
        selected_option = self.float_options[selected_index]
        disable_variables = ["variable1",
                             "variable2"]  # Fügen Sie hier die Variablennamen hinzu, die deaktiviert werden sollen

        for variable_name, line_edit, _ in self.input_fields:
            line_edit.setEnabled(variable_name not in disable_variables)

    def run_function(self):
        selected_function = self.function_combo.currentText()
        # Variablenwerte sammeln
        variables = {}
        for element in self.input_fields:
            if len(element) == 3:
                variable_name, line_edit, checkbox_default = element
                if variable_name != 'folderpath' and variable_name != 'show_plots':
                    if checkbox_default.isChecked() or line_edit.text() == '':
                        variable_value = None
                    else:
                        variable_value = float(line_edit.text())
                else:
                    if variable_name == 'show_plots':
                        if self.checkbox_yes.isChecked():
                            variable_value = True
                        elif self.checkbox_no.isChecked():
                            variable_value = False

                    else:
                        variable_value = line_edit.text()
                variables[variable_name] = variable_value
            elif len(element) == 2:
                variable_name, line_edit = element
                variable_value = line_edit.text()
                variables[variable_name] = variable_value
            else:
                variable_name, line_edit, _ = element
                variable_value = line_edit.text()
                variables[variable_name] = variable_value
        if self.unit_dva_checkbox.isChecked():
            variables['unit'] = 'DVA'
        elif self.unit_arcmin_checkbox.isChecked():
            variables['unit'] = 'Arcmin'
        elif self.unit_um_checkbox.isChecked():
            variables['unit'] = 'µm'

        current_combo = self.float_combo.currentText().split(',')
        for comb_elem in current_combo:
            comb_name=comb_elem.split('=')[0]
            comb_val = float(comb_elem.split('=')[1])
            variables.update({comb_name: comb_val})
        if selected_function == 'RandomWalk':
            if isinstance(variables['number'], type(None)):
                variables['number'] =1
            range_end = int(variables['number'])
            variables.pop('number')
            variables['potential_resolution'] = int(variables['grid size L'])
            variables.pop('grid size L')
            for i in range(1, range_end + 1):
                RandomWalk.RandomWalk.randomWalk(**variables, number_id=i)
        if selected_function == 'RCGAN':
            hyperparameterfile = ""
            model = GenerateDataset(hyperparameterfile)
            f_samp = variables['sampling_frequency']
            duration = variables['duration']
            n = variables['number']
            labels = variables['label_file']
            model.generate_data(model, n, duration, f_samp,
                            labels="GeneratingTraces_RGANtorch\FEM\RoordaLabels.csv")#TODO: LabelFile anpassen, dass aus Input genommen wird


app = QApplication(sys.argv)
window = MyWindow()
window.show()
sys.exit(app.exec_())
