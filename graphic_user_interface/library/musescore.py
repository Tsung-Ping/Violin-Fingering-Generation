import logging
import subprocess
from pathlib import Path
from tkinter import filedialog, messagebox

import music21

_music21_user_settings = music21.environment.UserSettings()


def _is_musescore_working(musescore_path):
    if not musescore_path.exists():
        return False

    cmd_check_app = [str(musescore_path), "--long-version"]
    try:
        # subprocess.run(cmd_check_app, shell=True, text=True, check=True)
        subprocess.run(cmd_check_app, shell=True, check=True)
    except subprocess.CalledProcessError:
        return False
    else:
        return True


def _get_musescore_from_music21_config():
    return Path(_music21_user_settings["musicxmlPath"]).resolve()


def _set_musescore_to_music21_config(musescore_path):
    _music21_user_settings["musicxmlPath"] = musescore_path


def _guess_musescore_path():
    guess_seed = Path("C:/Program Files/MuseScore 3/bin/MuseScore3.exe")
    while not guess_seed.exists() and guess_seed.parent != guess_seed:
        guess_seed = guess_seed.parent
    return guess_seed


def _ask_user_provide_musescore_path():
    messagebox.showwarning(
        "Need MuseScore3",
        "This GUI tool need MuseScore3 to render score preview\n"
        "Please provide a valid path of MuseScore3\n"
        "Ususally, musescore is located in C:/Program Files/MuseScore 3/bin/MuseScore3.exe",
    )
    logging.warning("please provide a valid path of MuseScore3 to music21 settings")

    filepath = filedialog.askopenfilename(
        initialdir=_guess_musescore_path(),
        title="Select MuseScore3 executable(.exe) file",
        filetypes=(
            ("musescore exe", "*.exe"),
            ("all files", "*.*"),
        ),
    )
    return Path(filepath).resolve()


#
# Public API
#


def get_musescore():
    p = _get_musescore_from_music21_config()
    if _is_musescore_working(p):
        return p
    logging.error("%s is not working", p)

    p = _guess_musescore_path()
    if _is_musescore_working(p):
        _set_musescore_to_music21_config(p)
        return p
    logging.error("%s is not working", p)

    p = _ask_user_provide_musescore_path()
    if _is_musescore_working(p):
        _set_musescore_to_music21_config(p)
        return p
    logging.error("%s is not working", p)

    raise RuntimeError("Need MuseScore3")
