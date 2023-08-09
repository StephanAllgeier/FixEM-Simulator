import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox, QLabel, QLineEdit, QPushButton, \
    QFileDialog, QCheckBox, QHBoxLayout, QButtonGroup

from GeneratingTraces_MathematicalModel import RandomWalkBased


class Functs:
    @staticmethod
    def getWalk():
        print("getWalk() wurde aufgerufen.")


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Hauptfenster-Einstellungen
        self.setWindowTitle("Funktionsauswahl")
        self.setGeometry(200, 200, 400, 400)

        # Haupt-Widget und Layout erstellen
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Funktionsauswahl-ComboBox erstellen
        self.function_combo = QComboBox(self)
        self.function_combo.addItem("RandomWalk based")
        self.function_combo.addItem("GAN based")
        self.layout.addWidget(self.function_combo)

        # Dropdown-Menü mit 16 Optionen erstellen
        self.float_options = [(20, 10, 0.1, 1.9), (50, 25, 0.002, 1.9), (50, 10, 0.001, 3.4), (50, 10, 0.002, 3.4),
                              (50, 10, 0.01, 3.4), (50, 10, 0.1, 3.4), (50, 10, 0.001, 3.9), (50, 10, 0.005, 3.9),
                              (50, 10, 0.01, 3.9), (100, 10, 0.001, 5.4), (100, 10, 0.002, 5.4), (100, 10, 0.01, 5.4),
                              (100, 10, 0.05, 5.4), (100, 10, 0.001, 5.9), (100, 10, 0.005, 5.9), (100, 10, 0.05, 6.9)]
        self.drop_var = ['simulation_freq', "potential_resolution", "relaxation_rate", "hc"]
        self.float_combo = QComboBox(self)
        for index, option in enumerate(self.float_options):
            self.float_combo.addItem(
                f"{self.drop_var[0]}={option[0]},{self.drop_var[1]}={option[1]},{self.drop_var[2]}={option[2]},{self.drop_var[3]}={option[3]}")
        self.layout.addWidget(self.float_combo)

        self.float_combo.currentIndexChanged.connect(self.toggle_input_fields_enabled)  # Event handler hinzufügen
        self.layout.addWidget(self.float_combo)
        # Zuordnung von Anzeigenamen zu Variablennamen
        self.variable_mapping = {"Number of simulations": "number",
                            "Duration": 'duration', "Field size in degree": 'field_size',
                            "Sampling Frequency": 'sampling_frequency',
                            'Number of step candidates': 'num_step_candidates', 'Folderpath to save to': 'folderpath',
                            'Show plots': 'show_plots'}
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
                checkbox_yes = QCheckBox("Yes")
                checkbox_no = QCheckBox("No")
                checkbox_yes.clicked.connect(lambda: self.handle_checkbox(checkbox_yes, checkbox_no))
                checkbox_no.clicked.connect(lambda: self.handle_checkbox(checkbox_no, checkbox_yes))
                self.input_fields.append((variable_name, checkbox_yes, checkbox_no))
                hbox = QHBoxLayout()
                hbox.addWidget(checkbox_yes)
                hbox.addWidget(checkbox_no)
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



        # Button zum Ausführen der Funktion erstellen
        self.run_button = QPushButton("Funktion ausführen", self)
        self.run_button.clicked.connect(self.run_function)
        self.layout.addWidget(self.run_button)

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Ordner auswählen")
        for variable_name, line_edit, _ in self.input_fields:
            if variable_name == "Variable 9":
                line_edit.setText(folder_path)

    def toggle_input_field(self, line_edit, checkbox_default):
        line_edit.setDisabled(checkbox_default.isChecked())

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
                    if line_edit.text() == 'Yes':
                        variable_value = True
                    elif line_edit.text == 'No':
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
        variables_dict = {}
        for key in variables.keys():
            if key in self.variable_mapping.keys():
                variables_dict[self.variable_mapping[key]] = variables[key]
        current_combo = self.float_combo.currentText().split(',')
        for comb_elem in current_combo:
            comb_name=comb_elem.split('=')[0]
            comb_val = float(comb_elem.split('=')[1])
            variables.update({comb_name: comb_val})
        if selected_function == 'RandomWalk based':
            if isinstance(variables['number'], type(None)):
                variables['number'] =1
            for i in range(1, int(variables['number']) + 1):
                RandomWalkBased.RandomWalk.randomWalk(**variables, number_id=i)
        if selected_function == 'GAN based':
            print('Not yet implemented.')


app = QApplication(sys.argv)
window = MyWindow()
window.show()
sys.exit(app.exec_())