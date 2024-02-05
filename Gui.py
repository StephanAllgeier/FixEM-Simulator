import ast
import json
import sys
from pathlib import Path
import pandas as pd

import openpyxl
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox, QLabel, QLineEdit, QPushButton, \
    QFileDialog, QCheckBox, QHBoxLayout, QButtonGroup, QMessageBox, QProgressDialog, QGroupBox

from GeneratingTraces_MathematicalModel import RandomWalk
from GeneratingTraces_RCGAN.FEM.generate_dataset import GenerateDataset

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

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.info_message = None
        self.exception=False
        # Hauptfenster-Einstellungen
        self.setWindowTitle("Create synthetic FEMs")
        self.setGeometry(200, 200, 600, 400)

        # Haupt-Widget und Layout erstellen
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)


        #Group Boxes erstellen
        self.group_box_random_walk = QGroupBox("Random Walk configuration")
        self.layout_RandomWalk = QVBoxLayout(self.group_box_random_walk)
        self.group_box_gan = QGroupBox('RCGAN configuration')
        self.layout_gan = QVBoxLayout(self.group_box_gan)

        # Funktionsauswahl-ComboBox erstellen
        self.function_combo = QComboBox(self)
        self.label_for_func = QLabel("Method:")
        self.layout.addWidget(self.label_for_func)
        self.function_combo.addItem("Random Walk")
        self.function_combo.addItem("RCGAN")
        self.layout.addWidget(self.function_combo)

        # Dropdown-Menü mit 16 Optionen erstellen
        self.float_options = []

        # Open Excelfile
        combinations = get_combination_from_csv(Path("ParameterInputGui.csv"))
        for comb in combinations:
            self.float_options.append((comb['simulation rate'], comb['cells per degree'], comb['relaxation rate'], comb['h_crit']))

        self.drop_var = ['simulation_freq', 'cells per deg', 'relaxation_rate', 'hc']
        self.float_combo = QComboBox(self)
        self.label_for_combo = QLabel("Parameter combination:")
        self.layout_RandomWalk.addWidget(self.label_for_combo)
        self.float_combo.addItem('None')
        for index, option in enumerate(self.float_options):
            self.float_combo.addItem(
                f"{self.drop_var[0]}={option[0]},{self.drop_var[1]}={option[1]},{self.drop_var[2]}={option[2]},{self.drop_var[3]}={option[3]}")
        self.layout_RandomWalk.addWidget(self.float_combo)
        self.disable_fields=False
        self.float_combo.currentIndexChanged.connect(self.toggle_input_fields_enabled)  # Event handler hinzufügen
        self.layout_RandomWalk.addWidget(self.float_combo)

        self.gan_model_file = QLabel("GAN Model file:")
        self.gan_model_edit = QLineEdit()
        self.gan_model_edit.setReadOnly(True)
        self.gan_model_browse_button = QPushButton("Browse file")

        self.label_file = QLabel("Label file:")
        self.label_file_edit = QLineEdit()
        self.label_file_edit.setReadOnly(True)
        self.label_file_browse_button = QPushButton("Browse file")

        self.gan_config_file = QLabel("GAN config file:")
        self.gan_config_edit = QLineEdit()
        self.gan_config_edit.setReadOnly(True)
        self.gan_config_browse_button = QPushButton("Browse file")

        self.label_simulation_freq = QLabel("Simulation Frequency in Hz (as float):")
        self.label_cells_per_degree = QLabel("Cells per Degree (as int):")
        self.label_relaxation_rate = QLabel("Relaxation Rate (as float):")
        self.label_field_size = QLabel("Field size in degree (as float)")
        self.label_hc = QLabel("hc (as float):")
        self.label_chi = QLabel("Angular weight chi (as float):")

        self.layout_RandomWalk.addWidget(self.label_simulation_freq)
        self.line_edit_simulation_freq = QLineEdit()
        self.layout_RandomWalk.addWidget(self.line_edit_simulation_freq)

        self.layout_RandomWalk.addWidget(self.label_cells_per_degree)
        self.line_edit_cells_per_degree = QLineEdit()
        self.layout_RandomWalk.addWidget(self.line_edit_cells_per_degree)

        self.layout_RandomWalk.addWidget(self.label_relaxation_rate)
        self.line_edit_relaxation_rate = QLineEdit()
        self.layout_RandomWalk.addWidget(self.line_edit_relaxation_rate)

        self.layout_RandomWalk.addWidget(self.label_hc)
        self.line_edit_hc = QLineEdit()
        self.layout_RandomWalk.addWidget(self.line_edit_hc)

        self.layout_RandomWalk.addWidget(self.label_field_size)
        self.line_edit_field_size = QLineEdit()
        self.layout_RandomWalk.addWidget(self.line_edit_field_size)

        self.layout_RandomWalk.addWidget(self.label_chi)
        self.line_edit_chi = QLineEdit()
        self.layout_RandomWalk.addWidget(self.line_edit_chi)

        self.layout.addWidget(self.group_box_random_walk)
        '''
        # input_fields-Liste     aktualisieren
        self.input_fields = [
            ("simulation_freq", self.line_edit_simulation_freq, None),
            ("cells per degree", self.line_edit_cells_per_degree, None),
            ("relaxation_rate", self.line_edit_relaxation_rate, None),
            ("hc", self.line_edit_hc, None),
            ("chi", self.line_edit_chi, None)
        ]
        '''
        self.layout_gan.addWidget(self.gan_model_file)
        self.layout_gan.addWidget(self.gan_model_edit)
        self.layout_gan.addWidget(self.gan_model_browse_button)
        self.gan_model_browse_button.clicked.connect(lambda: self.update_file_edit(self.gan_model_edit, "GAN model file (*.pth)"))

        self.layout_gan.addWidget(self.label_file)
        self.layout_gan.addWidget(self.label_file_edit)
        self.layout_gan.addWidget(self.label_file_browse_button)
        self.label_file_browse_button.clicked.connect(lambda: self.update_file_edit(self.label_file_edit, "Label files (*.csv)"))

        self.layout_gan.addWidget(self.gan_config_file)
        self.layout_gan.addWidget(self.gan_config_edit)
        self.layout_gan.addWidget(self.gan_config_browse_button)
        self.gan_config_browse_button.clicked.connect(lambda: self.update_file_edit(self.gan_config_edit, "Config files (*.json)"))

        self.layout.addWidget(self.group_box_gan)
        # Zuordnung von Anzeigenamen zu Variablennamen
        self.variable_mapping = {
                            "Number of simulations": "number",
                            "Duration in seconds (as int)": 'duration',
                            #"Field size in degree (as float)": 'field_size',
                            "Sampling Frequency in Hz (as float)": 'sampling_frequency',
                            'Folderpath to save to': 'folderpath'
                            #'Show plots': 'show_plots'
                            }
        # Felder für Variableneingabe erstellen
        self.input_fields = []
        self.default_checkboxes = []
        self.variable10_checkbox_group = QButtonGroup(self)  # ButtonGroup für Variable 10 Checkboxen
        self.variable10_checkbox_group.setExclusive(True)  # Nur eine Checkbox kann ausgewählt sein
        for display_name, variable_name in self.variable_mapping.items():
            if display_name == "Folderpath to save to":
                label = QLabel(display_name + ":")
                line_edit = QLineEdit()
                line_edit.setReadOnly(True)
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

        #self.checkbox_no.setChecked(True)

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

        # Hinzufügen zum Haupt-Layout
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
        self.gan_model_edit.setText(file_path)

    def toggle_input_field(self, line_edit, checkbox_default):
        line_edit.setDisabled(checkbox_default.isChecked())
    def toggle_file_input_enabled(self):
        selected_function = self.function_combo.currentText()
        enable_file_input = selected_function == "RCGAN"

        self.gan_model_edit.setEnabled(enable_file_input)
        self.gan_model_browse_button.setEnabled(enable_file_input)
        self.label_file_edit.setEnabled(enable_file_input)
        self.label_file_browse_button.setEnabled(enable_file_input)
        self.gan_config_edit.setEnabled(enable_file_input)
        self.gan_config_browse_button.setEnabled(enable_file_input)
        self.float_combo.setEnabled(not enable_file_input)
        for label, line_edit in [
            (self.label_simulation_freq, self.line_edit_simulation_freq),
            (self.label_cells_per_degree, self.line_edit_cells_per_degree),
            (self.label_relaxation_rate, self.line_edit_relaxation_rate),
            (self.label_hc, self.line_edit_hc),
            (self.label_chi, self.line_edit_chi),
            (self.label_field_size, self.line_edit_field_size)
        ]:
            label.setEnabled(not enable_file_input)
            line_edit.setDisabled(enable_file_input)
            line_edit.clear() if not enable_file_input else None

    def handle_checkbox(self, checkbox, other_checkbox):
        if checkbox.isChecked():
            other_checkbox.setChecked(False)

    def toggle_input_fields_enabled(self):
        selected_option_text = self.float_combo.currentText()
        self.disable_fields = selected_option_text.lower() != 'none'

        for label, line_edit in [
            (self.label_simulation_freq, self.line_edit_simulation_freq),
            (self.label_cells_per_degree, self.line_edit_cells_per_degree),
            (self.label_relaxation_rate, self.line_edit_relaxation_rate),
            (self.label_hc, self.line_edit_hc)
        ]:
            label.setEnabled(not self.disable_fields)
            line_edit.setDisabled(self.disable_fields)
            line_edit.clear() if self.disable_fields else None

    def update_file_edit(self, file_edit, file_filter):
        file_name, _ = QFileDialog.getOpenFileName(self, "Datei auswählen", "", file_filter)
        if file_name:
            file_edit.setText(file_name)
    def load_config(self, file_path):
        with open(file_path, 'r') as file:
            config_data = json.load(file)
        return config_data

    def run_function(self):
        self.exception = False
        selected_function = self.function_combo.currentText()
        if selected_function == "RCGAN":
            if self.gan_model_edit.text() == '':
                QMessageBox.warning(self, 'Warning', f"Please enter a valid model file.")
                self.reset_gui()
            elif self.gan_config_edit.text() == '':
                QMessageBox.warning(self, 'Warning', f"Please enter a valid config file.")
                self.reset_gui()
            elif self.label_file_edit.text() == '':
                QMessageBox.warning(self, 'Warning', f"Please enter a valid label file.")
                self.reset_gui()

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
        if variables['folderpath'] == '':
            QMessageBox.warning(self, 'Warning','Please enter a folder to save to...')
            return
        if self.unit_dva_checkbox.isChecked():
            variables['unit'] = 'DVA'
        elif self.unit_arcmin_checkbox.isChecked():
            variables['unit'] = 'Arcmin'
        elif self.unit_um_checkbox.isChecked():
            variables['unit'] = 'µm'


        current_combo = self.float_combo.currentText().split(',')
        for comb_elem in current_combo:
            if comb_elem == 'None':
                break
            comb_name=comb_elem.split('=')[0]
            comb_val = float(comb_elem.split('=')[1])
            variables.update({comb_name: comb_val})
        if not self.disable_fields:
            def get_float_input(line_edit, name):
                text = line_edit.text()
                if text == '':
                    return None
                else:
                    try:
                        return_val = float(text)
                        return return_val
                    except Exception:
                        QMessageBox.critical(self, 'Error', f"{name} is expected to be float.")
                        self.exception =True
            variables["simulation_freq"] = get_float_input(self.line_edit_simulation_freq, 'Simulation frequency')
            variables["cells per deg"] = get_float_input(self.line_edit_cells_per_degree, 'Cells per degree')
            variables["relaxation_rate"] = get_float_input(self.line_edit_relaxation_rate, 'Relaxation rate')
            variables["hc"] = get_float_input(self.line_edit_hc, 'h_crit')
            variables["field_size"] = get_float_input(self.line_edit_field_size, 'field_size')
            if self.line_edit_chi.text() != '':
                try:
                    variables['chi'] = float(self.line_edit_chi.text())
                except Exception as e:
                    QMessageBox.critical(self, 'Error', f"Chi is expected to be float.")
                    self.exception = True
            else:
                variables['chi'] = 1
        if selected_function == 'Random Walk':
            if isinstance(variables['number'], type(None)):
                variables['number'] =1
            range_end = int(variables['number'])
            variables.pop('number')
            if variables['cells per deg'] != '' and variables['cells per deg'] != None:
                variables['potential_resolution'] = int(variables['cells per deg'])
            else:
                variables['potential_resolution'] = None
            variables.pop('cells per deg')
            # ProgressBar
            if not self.exception:
                progress_dialog = QProgressDialog("Simulations running", "Cancel", 0, range_end, self)
                progress_dialog.setWindowTitle("Progress")
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.setAutoClose(True)

                for i in range(1, range_end + 1):
                    if progress_dialog.wasCanceled():
                        break
                    progress_dialog.setValue(i-1)
                    QApplication.processEvents()

                    RandomWalk.RandomWalk.randomWalk(**variables, number_id=i)
                progress_dialog.setValue(range_end)
                if not progress_dialog.wasCanceled():
                    self.show_info_popup(variables, range_end)
                    self.wait_for_popup_close()
        if selected_function == 'RCGAN':
            try:
                hyperparameterfile = self.gan_model_edit.text()
                model = GenerateDataset(hyperparameterfile)
                config_data = self.load_config(self.gan_config_edit.text())
                f_sampling = variables['sampling_frequency']
                duration = variables['duration']
                n = variables['number']
                labels = self.label_file_edit.text()
                model.generate_data(model.model, n, duration, fsamp=config_data['resample_freq'], fsamp_out=f_sampling, folderpath_to_save_to=variables['folderpath'],
                                labels=labels, noise_scale=config_data['scale'], unit=variables['unit'])
                self.show_info_popup(variables, range_end)
                self.wait_for_popup_close()
            except Exception:
                QMessageBox.critical(self, 'Error', f"There was an error generating traces. Probably linked to false input files.")
        self.reset_gui()

    def reset_gui(self):

        self.function_combo.setCurrentIndex(0)
        self.float_combo.setCurrentIndex(0)
        self.gan_model_edit.clear()
        self.label_file_edit.clear()
        self.gan_config_edit.clear()
        self.line_edit_simulation_freq.clear()
        self.line_edit_cells_per_degree.clear()
        self.line_edit_relaxation_rate.clear()
        self.line_edit_hc.clear()
        self.line_edit_chi.clear()

        # Deaktiviere alle Checkboxen
        for checkbox in self.default_checkboxes:
            checkbox.setChecked(False)

        self.unit_dva_checkbox.setChecked(True)

        # Deaktiviere Eingabefelder
        self.toggle_input_fields_enabled()
        self.toggle_file_input_enabled()

    def show_info_popup(self, variables, number):
        self.info_message = QMessageBox(self)
        self.info_message.setIcon(QMessageBox.Information)
        self.info_message.setWindowTitle("Simulations done")
        self.info_message.setText("All simulations have run successfully.")
        self.info_message.setDetailedText(
            f"Savepath: {variables['folderpath']}\nNumber of Simulations: {number}\nSimulation Duration: {variables['duration']}")
        self.info_message.open()

    def wait_for_popup_close(self):
        popup_open = True
        while popup_open:
            QApplication.processEvents()
            popup_open = self.info_message.isVisible()


app = QApplication(sys.argv)

while True:
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
    break