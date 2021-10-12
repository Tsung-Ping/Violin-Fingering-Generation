import csv
import enum
import logging
import subprocess
import tempfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pretty_midi
from inference import read_csv
from lxml import etree
from music21 import converter
from violin_fingering_model import violin_fingering_model

from library.musescore import get_musescore

NOT_APPLICABLE = "N/A"


class PreferenceMode(enum.Enum):
    BASIC_MODE = "basic"
    LOWEST_MODE = "lowest"
    NEAREST_MODE = "nearest"


class Strings(enum.IntEnum):
    N = 0
    G = 1
    D = 2
    A = 3
    E = 4


position_tamplate = """<direction placement="above">
            <direction-type>
                <words relative-y="20.00">{position}</words>
            </direction-type>
        </direction>"""

string_tamplate = """<direction placement="above">
        <direction-type>
          <words relative-x="-1.81" relative-y="45.12">Sul {string}</words>
          </direction-type>
        </direction>"""

notations_sample = """<notations>
            <technical>
                <fingering default-x="2" default-y="31" placement="above">{finger}</fingering>
                <!--down-bow default-x="0" default-y="10" placement="above"/-->
            </technical>
        </notations>"""


class _Parser:
    def __init__(self, musescore_path):
        self.musescore_path = musescore_path

    @property
    def output_xmltree(self):
        if not hasattr(self, "_output_xmltree"):
            self.reset_output_xmltree()
        return self._output_xmltree

    @output_xmltree.setter
    def output_xmltree(self, output_xmltree):
        self._output_xmltree = output_xmltree

    @property
    def output_notes(self):
        if not hasattr(self, "_output_notes"):
            self.reset_output_notes()
        return self._output_notes

    @output_notes.setter
    def output_notes(self, output_notes):
        self._output_notes = output_notes

    @property
    def score(self):
        if not hasattr(self, "_score"):
            raise RuntimeError("No score property, please load_musicxml first")
        return self._score

    @score.setter
    def score(self, score):
        self._score = score

    @property
    def import_notes(self):
        if not hasattr(self, "_import_notes"):
            raise RuntimeError("No import notes, please import csv first")
        return self._import_notes

    @import_notes.setter
    def import_notes(self, import_notes):
        self._import_notes = import_notes

    @staticmethod
    def write_dict_csv(notes, csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(notes[0].keys()))
            writer.writeheader()
            for note in notes:
                writer.writerow(note)

    @staticmethod
    def read_dict_csv(csv_path):
        notes_list = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter=",")
            for row in reader:
                notes_list.append(row)
        return notes_list

    def _update_output_notes(self, notes):
        for index, note in enumerate(notes):
            # those property should be overwrite
            for note_property in ["string", "position", "finger"]:
                self.output_notes[index][note_property] = note[note_property]

    def _update_output_xmltree(self, notes):
        output_xmlnotes = self.output_xmltree.getroot().findall(".//note")
        previous_string = None

        for index, note in enumerate(notes):
            pred_finger = note["finger"]
            if pred_finger != NOT_APPLICABLE:
                finger_element = etree.XML(notations_sample.format(finger=pred_finger))
                output_xmlnotes[index].append(finger_element)

            pred_position = note["position"]
            UPDATE_POSITION = False
            if UPDATE_POSITION and pred_position != NOT_APPLICABLE:
                position_element = etree.XML(
                    position_tamplate.format(position=pred_position)
                )
                output_xmlnotes[index].addprevious(position_element)

            # Only show string when it changes.
            pred_string = (
                Strings[note["string"]].name
                if note["string"] != NOT_APPLICABLE
                else note["string"]
            )
            if pred_string != NOT_APPLICABLE and pred_string != previous_string:
                string_element = etree.XML(string_tamplate.format(string=pred_string))
                output_xmlnotes[index].addprevious(string_element)
                previous_string = pred_string

    def _verify_import_notes(self, import_notes):
        assert len(import_notes) == len(
            self.output_notes
        ), "the total number of notes from import csv should be same as original xml"
        for index, note in enumerate(import_notes):
            # those property should not be change
            for note_property in [
                "pitch",
                "time_start",
                "duration",
                "beat_type",
            ]:
                assert (
                    self.output_notes[index][note_property] == note[note_property]
                ), f"{self.output_notes[index][note_property]} != {note[note_property]} note property:{note_property} from import csv should be same as original xml"

    @staticmethod
    def _convert_beat_type_naming(beat_type):
        beat_type_naming_lookup = {
            "eighth": "8th",
            "quarter": "16th",
            "32nd": "32th",
        }
        return beat_type_naming_lookup.get(beat_type, beat_type)

    def _get_music_property(self, index, note):
        xml_note = self.xmlnotes[index]
        xml_duration = xml_note.find("duration").text
        music_property = {
            "pitch": note.pitch.midi,
            "time_start": int((float(note.beat) - 1) * self.divisions),
            "duration": int(xml_duration),
            "beat_type": self._convert_beat_type_naming(note.duration.type),
            "string": NOT_APPLICABLE,
            "position": NOT_APPLICABLE,
            "finger": NOT_APPLICABLE,
        }
        # make all value string
        music_property = {key: str(value) for key, value in music_property.items()}
        return music_property

    def _get_notes_from_source(self, score):
        notes_list = []
        for index, note in enumerate(score.recurse().notes):
            if note.isNote:
                music_property = self._get_music_property(index, note)
            elif note.isChord:
                for n in note._notes:
                    music_property = self._get_music_property(index, n)
            notes_list.append(music_property)
        return notes_list

    @staticmethod
    def _convert_beat_type(beat_type):
        beat_type_lookup = {
            "": 0,
            "1th": 1,
            "2th": 2,
            "4th": 3,
            "8th": 4,
            "16th": 5,
            "32th": 6,
        }
        if beat_type in beat_type_lookup.keys():
            for name, index in beat_type_lookup.items():
                if name == beat_type:
                    return index
            else:
                raise LookupError("Unknown beat type")
        elif beat_type in beat_type_lookup.values():
            for name, index in beat_type_lookup.items():
                if index == beat_type:
                    return name
            else:
                raise LookupError("Unknown beat type")
        else:
            raise ValueError("Cannot convert beat type")

    def _convert_list_to_notes(
        self,
        pitches,
        starts,
        durations,
        beat_types,
        pred_string,
        pred_position,
        pred_finger,
    ):
        notes_list = []
        n_notes = len(pitches)
        for i in range(n_notes):
            music_property = {
                "pitch": pitches[i],
                "time_start": starts[i],
                "duration": durations[i],
                "beat_type": self._convert_beat_type(beat_types[i]),
                "string": pred_string[i],
                "position": pred_position[i],
                "finger": pred_finger[i],
            }
            music_property = {key: str(value) for key, value in music_property.items()}
            notes_list.append(music_property)
        self._verify_import_notes(notes_list)
        return notes_list

    @staticmethod
    def _show_result_by_pretty_midi(pitches, pred_string, pred_position, pred_finger):
        print(
            "pitch".ljust(9),
            "".join(
                [pretty_midi.note_number_to_name(number).rjust(4) for number in pitches]
            ),
        )
        print(
            "string".ljust(9),
            "".join([s.rjust(4) for s in pred_string]),
        )
        print(
            "position".ljust(9),
            "".join([p.rjust(4) for p in pred_position]),
        )
        print(
            "finger".ljust(9),
            "".join([f.rjust(4) for f in pred_finger]),
        )


#
# Public API
#
class Parser(_Parser):
    def is_musicxml_loaded(self):
        return hasattr(self, "_score")

    def is_import_loaded(self):
        return hasattr(self, "_import_notes")

    def save_musicxml(self, xml_path):
        logging.debug("saving musicxml to %s", xml_path)

        with open(xml_path, "wb") as f:
            self.output_xmltree.write(
                f,
                xml_declaration=True,
                encoding=self.output_xmltree.docinfo.encoding,
                standalone=self.output_xmltree.docinfo.standalone,
                method="xml",
            )
        logging.debug("XML created successfully")

    def save_as_pdf(self, pdf_path):
        logging.debug("saving pdf to %s", pdf_path)
        # Save temp musicxml and convert to pdf
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_xml = Path(tmpdir, "temp_for_convert_to_pdf").with_suffix(".xml")
            self.save_musicxml(temp_xml)
            cmd_convert_pdf = list(
                map(str, [self.musescore_path, "-o", pdf_path, temp_xml])
            )
            subprocess.run(cmd_convert_pdf, shell=True, text=True, check=True)
        logging.debug("PDF created successfully")

    def save_as_csv(self, csv_path):
        logging.debug("saving CSV to %s", csv_path)
        self.write_dict_csv(self.output_notes, csv_path)
        logging.debug("CSV created successfully")

    def load_musicxml(self, filepath):
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} is not exists")

        if filepath.suffix not in [".xml", ".musicxml"]:
            raise TypeError(f"{filepath.suffix} is not supported")

        xmlparser = etree.XMLParser(dtd_validation=False)
        try:
            self.xmltree = etree.parse(str(filepath), xmlparser)
            self.xmlnotes = self.xmltree.getroot().findall(".//note")
            self.divisions = float(
                self.xmltree.find("./part/measure/attributes/divisions").text
            )
        except Exception:
            logging.error("%s is not a valid XML or can not parse", filepath.name)
            raise

        try:
            self.score = converter.parse(filepath, format="musicxml").flat
            self.source_notes = self._get_notes_from_source(self.score)
        except:
            logging.error("converting musicxml to notes fail")
            raise

        logging.debug("load musicxml successfully")
        self.temp_notes_path = filepath.parent / "temp_notes.csv"
        self.save_as_csv(self.temp_notes_path)
        return self.output_notes

    def reset_output_notes(self):
        self.output_notes = deepcopy(self.source_notes)

    def reset_output_xmltree(self):
        self.output_xmltree = deepcopy(self.xmltree)

    def load_import(self, filepath):
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} is not exists")

        if filepath.suffix not in [".csv"]:
            raise TypeError(f"{filepath.suffix} is not supported")

        try:
            self.import_notes = self.read_dict_csv(filepath)
            self._verify_import_notes(self.import_notes)
        except Exception:
            logging.error("Something wrong happened during import %s", filepath.name)
            raise

        self.last_import = filepath
        logging.debug("get notes from import successfully")
        self.show_csv()
        return self.import_notes

    def show_csv(self):
        self._update_output_notes(self.import_notes)
        self._update_output_xmltree(self.output_notes)

    def recommend(self, mode=PreferenceMode.BASIC_MODE):
        assert isinstance(mode, PreferenceMode)
        input = read_csv(self.temp_notes_path)
        pitches, starts, durations, beat_types, strings, positions, fingers = (
            input["pitches"],
            input["starts"],
            input["durations"],
            input["beat_types"],
            input["strings"],
            input["positions"],
            input["fingers"],
        )
        model = violin_fingering_model()
        pred_str, pred_pos, pred_fin = model.inference(
            pitches=pitches,
            starts=starts,
            durations=durations,
            beat_types=beat_types,
            strings=strings,
            positions=positions,
            fingers=fingers,
            mode=mode.value,
        )

        # convert_ndarray_to_list
        n_notes = len(pitches)
        pred_string = [Strings(s).name for s in pred_str[0, :n_notes]]
        pred_position = [str(p) for p in pred_pos[0, :n_notes]]
        pred_finger = [str(f) for f in pred_fin[0, :n_notes]]
        self._show_result_by_pretty_midi(
            pitches, pred_string, pred_position, pred_finger
        )

        recommend_notes = self._convert_list_to_notes(
            pitches,
            starts,
            durations,
            beat_types,
            pred_string,
            pred_position,
            pred_finger,
        )
        logging.debug("get notes from recommend successfully")
        self._update_output_notes(recommend_notes)
        self._update_output_xmltree(self.output_notes)

    def get_preview_images(self):
        # Save temp musicxml and Convert to image png for preview
        temp_xml = Path(tempfile.mkdtemp(), "temp_for_preview").with_suffix(".xml")
        self.save_musicxml(temp_xml)

        output_png = temp_xml.with_name(f"{temp_xml.stem}.png")
        cmd_convert_png = list(
            map(str, [self.musescore_path, "-o", output_png, temp_xml])
        )
        # subprocess.run(cmd_convert_png, shell=True, text=True, check=True)
        subprocess.run(cmd_convert_png, shell=True, check=True)
        preview_images = list(output_png.parent.glob(f"{output_png.stem}-*.png"))
        return preview_images
