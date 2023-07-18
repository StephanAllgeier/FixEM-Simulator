import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox, QLabel, QLineEdit, QPushButton, \
    QFileDialog, QCheckBox, QHBoxLayout
from GeneratingTraces_MathematicalModel import RandomWalkBased

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Hauptfenster-Einstellungen
        self.setWindowTitle("Funktionsauswahl")
        self.setGeometry(200, 200, 400, 300)

        # Haupt-Widget und Layout erstellen
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Funktionsauswahl-ComboBox erstellen
        self.function_combo = QComboBox(self)
        self.function_combo.addItem("RandomWalk based")
        self.function_combo.addItem("GAN based")
        self.layout.addWidget(self.function_combo)

        # Liste von Variablennamen erstellen
        self.variable_names = ["Duration", "Field size in degree", "Simulation Frequency", "Sampling Frequency", "Sampling duration",
                               "Relaxation Rate", "Start Position Sigma", 'Number of step candidates', 'Folderpath to save to', 'Show plots', 'Cells per degree']
        self.variable_mapping = {"Cells per degree": "potential_resolution", "Duration": 'duration', "Field size in degree":'field_size', "Simulation Frequency":'simulation_freq', "Sampling Frequency":'sampling_frequency', "Sampling duration":'sampling_duration',
                               "Relaxation Rate":'relaxation_rate', "Start Position Sigma":'start_position_sigma', 'Number of step candidates':'num_step_candidates', 'Folderpath to save to':'folderpath', 'Show plots':'show_plots'}

        self.input_fields = []
        for variable_name in self.variable_names:
            if variable_name == "Folderpath to save to":
                label = QLabel(variable_name + ":")
                line_edit = QLineEdit()
                browse_button = QPushButton("Browse folder")
                browse_button.clicked.connect(self.browse_folder)
                self.input_fields.append((variable_name, line_edit, browse_button))
                self.layout.addWidget(label)
                self.layout.addWidget(line_edit)
                self.layout.addWidget(browse_button)
            elif variable_name == "Show plots":
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
                label = QLabel(variable_name + ":")
                line_edit = QLineEdit()
                checkbox_default = QCheckBox("Default")
                checkbox_default.clicked.connect(
                    lambda state, line_edit=line_edit: self.toggle_input_field(line_edit, state))
                self.input_fields.append((variable_name, line_edit, checkbox_default))
                hbox = QHBoxLayout()
                hbox.addWidget(line_edit)
                hbox.addWidget(checkbox_default)
                self.layout.addWidget(label)
                self.layout.addLayout(hbox)

        # Button zum Ausführen der Funktion erstellen
        self.run_button = QPushButton("Funktion ausführen", self)
        self.run_button.clicked.connect(self.run_function)
        self.layout.addWidget(self.run_button)

    def handle_checkbox(self, clicked_checkbox, other_checkbox):
        if clicked_checkbox.isChecked():
            other_checkbox.setChecked(False)

    def toggle_input_field(self, line_edit, checkbox_default):
        line_edit.setDisabled(checkbox_default)

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Ordner auswählen")
        for element in self.input_fields:
            if len(element) == 3:
                variable_name, line_edit, _ = element
                if variable_name == "Folderpath to save to":
                    line_edit.setText(folder_path)


    def run_function(self):
        selected_function = self.function_combo.currentText()

        # Variablenwerte sammeln
        variables = {}
        for element in self.input_fields:
            if len(element) == 3:
                variable_name, line_edit, checkbox_default = element
                if variable_name != 'Folderpath to save to' and variable_name != 'Show plots':
                    if checkbox_default.isChecked() or line_edit.text() == '':
                        variable_value=None
                    else:
                        variable_value = float(line_edit.text())
                else:
                    if line_edit.text() == 'Yes':
                        variable_value=True
                    elif line_edit.text == 'No':
                        variable_value=False
                    else:
                        variable_value=line_edit.text()
                variables[variable_name] = variable_value
            elif len(element) == 2:
                variable_name, line_edit = element
                variable_value = line_edit.text()
                variables[variable_name] = variable_value
            else:
                variable_name, line_edit,_ = element
                variable_value = line_edit.text()
                variables[variable_name] = variable_value
        variables_dict={}
        for key in variables.keys():
            if key in self.variable_mapping.keys():
                variables_dict[self.variable_mapping[key]] = variables[key]
        if selected_function == 'RandomWalk based':
            RandomWalkBased.RandomWalk.randomWalk(**variables_dict)
        if selected_function == 'GAN based':
            print('Not yet implemented.')

        # Hier können Sie den Code einfügen, um die ausgewählte Funktion mit den gegebenen Variablen auszuführen
        # und das Ergebnis anzuzeigen
        # Beispiel:
        print("Ausgewählte Funktion:", selected_function)
        print("Eingegebene Variablen:", variables)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())