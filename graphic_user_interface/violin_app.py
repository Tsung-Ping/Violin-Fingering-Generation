import logging
import tkinter as tk
import tkinter.ttk as ttk
from collections import defaultdict
from pathlib import Path
from tkinter import filedialog

from PIL import Image, ImageTk

from library.musescore import get_musescore
from library.parser import Parser, PreferenceMode

class ViolinApp(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.initialdir = defaultdict(Path.cwd)
        self._init_main_window()
        self._init_main_window_bindings()
        self._init_gui_layout()
        self.parser = Parser(get_musescore())

    def _resize_image(self, width, height):
        size = (max(width - 200, 30), height)
        resized = self.original_image.resize(size, Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete("IMG")
        self.preview_canvas.create_image(
            0, 0, image=self.image, anchor=tk.N + tk.W, tags="IMG"
        )

    def _resize_event(self, event):
        self._resize_image(event.width, event.height)

    def _init_main_window_bindings(self):
        self.master.bind("<l>", lambda e: self.click_load())
        self.master.bind("<i>", lambda e: self.click_import())
        self.master.bind("<r>", lambda e: self.click_recommend())
        self.master.bind("<v>", lambda e: self.click_show_csv())
        self.master.bind("<c>", lambda e: self.click_clear())
        self.master.bind("<s>", lambda e: self.click_save())
        self.master.bind("<p>", lambda e: self.click_prev())
        self.master.bind("<n>", lambda e: self.click_next())
        self.master.bind("<Escape>", lambda e: self.master.destroy())
        self.bind("<Configure>", self._resize_event)

    def _init_main_window(self):
        screenwidth = self.master.winfo_screenwidth()
        screenheight = self.master.winfo_screenheight()
        width = screenwidth * 3 / 4
        height = screenheight * 3 / 4
        size = "%dx%d+%d+%d" % (
            width,
            height,
            (screenwidth - width) / 2,
            (screenheight - height) / 2,
        )
        self.master.geometry(size)
        self.master.title("Violin recommendation system")

    def _update_initialdir(self, name, filepath):
        self.initialdir[name] = Path(filepath).parent

    def _init_gui_layout(self):
        # left side preview canvas
        self.original_image = Image.open(
            Path(__file__).parent / "resources" / "onboard.jpg"
        )
        self.image = ImageTk.PhotoImage(self.original_image)
        self.preview_canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self.preview_canvas.create_image(
            0, 0, image=self.image, anchor=tk.N + tk.W, tags="IMG"
        )
        self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # right side control frame
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        control_element_layout = dict(
            fill=tk.BOTH,
            side=tk.TOP,
            expand=True,
            padx=5,
            pady=5,
            ipadx=10,
            ipady=10,
        )

        # load
        self.button_load = tk.Button(
            control_frame, text="Load (l)", command=self.click_load
        )
        self.button_load.pack(**control_element_layout)

        # import
        self.button_import = tk.Button(
            control_frame, text="Import (i)", command=self.click_import
        )
        self.button_import.pack(**control_element_layout)

        # mode
        self.combobox_mode = ttk.Combobox(
            control_frame,
            justify="center",
            state="readonly",
            values=[mode.name for mode in PreferenceMode],
        )
        self.combobox_mode.current(0)
        self.combobox_mode.pack(**control_element_layout)

        # recommend
        self.button_recommend = tk.Button(
            control_frame, text="Recommend (r)", command=self.click_recommend
        )
        self.button_recommend.pack(**control_element_layout)

        # show csv
        self.button_show_csv = tk.Button(
            control_frame, text="Show CSV (v)", command=self.click_show_csv
        )
        self.button_show_csv.pack(**control_element_layout)

        # clear
        self.button_clear = tk.Button(
            control_frame, text="Clear (c)", command=self.click_clear
        )
        self.button_clear.pack(**control_element_layout)

        # save
        self.button_save = tk.Button(
            control_frame, text="Save (s)", command=self.click_save
        )
        self.button_save.pack(**control_element_layout)

        # Canvas control
        canvas_control = tk.Frame(control_frame)
        canvas_control.pack(**control_element_layout)

        canvas_control_layout = control_element_layout.copy()
        canvas_control_layout.update(side=tk.LEFT)

        # prev
        self.button_prev = tk.Button(
            canvas_control, text="Prev (p)", command=self.click_prev
        )
        self.button_prev.pack(**canvas_control_layout)

        # next
        self.button_next = tk.Button(
            canvas_control, text="Next (n)", command=self.click_next
        )
        self.button_next.pack(**canvas_control_layout)

    def click_load(self):
        filepath = filedialog.askopenfilename(
            initialdir=self.initialdir["load"],
            title="Select Musicxml file",
            filetypes=(
                ("musicxml files", "*.musicxml *.xml"),
                ("all files", "*.*"),
            ),
        )
        if not filepath:
            return
        self._update_initialdir("load", filepath)
        self.parser.load_musicxml(filepath)

        self._reset_preview_preload()
        self._resize_image(self.winfo_width(), self.winfo_height())

    def click_import(self):
        filepath = filedialog.askopenfilename(
            initialdir=self.initialdir["import"],
            title="Select CSV file",
            filetypes=(
                ("csv files", "*.csv"),
                ("all files", "*.*"),
            ),
        )
        if not filepath:
            return
        self._update_initialdir("import", filepath)

        self.parser.load_import(filepath)

        self._reset_preview_preload()
        self._resize_image(self.winfo_width(), self.winfo_height())

    def click_recommend(self):
        if not self.parser.is_musicxml_loaded():
            logging.error("please load musicxml first")
            return

        if self.parser.is_import_loaded():
            logging.debug("user import is loaded: %s", self.parser.last_import)
        mode = PreferenceMode[self.combobox_mode.get()]
        logging.debug("mode selected: %s", mode)

        self.parser.recommend(mode)

        self._reset_preview_preload()
        self._resize_image(self.winfo_width(), self.winfo_height())

    def _reset_preview_preload(self):
        self.preview_preload = list(map(Image.open, self.parser.get_preview_images()))
        self.preview_index = 0
        self.preview_count = len(self.preview_preload)
        self.original_image = self.preview_preload[self.preview_index]

    def click_show_csv(self):
        if not self.parser.is_import_loaded():
            logging.error("please import first")
            return
        self.parser.show_csv()
        self._reset_preview_preload()
        self._resize_image(self.winfo_width(), self.winfo_height())

    def click_clear(self):
        self.parser.reset_output_notes()
        self.parser.reset_output_xmltree()
        self._reset_preview_preload()
        self._resize_image(self.winfo_width(), self.winfo_height())

    def click_save(self):
        if not self.parser.is_musicxml_loaded():
            logging.error("please load musicxml first")
            return

        filepath = filedialog.asksaveasfilename(
            initialdir=self.initialdir["save"],
            title="Save as XML, PDF, or CSV files",
            filetypes=(
                ("musicxml files", "*.musicxml *.xml"),
                ("pdf files", "*.pdf"),
                ("csv files", "*.csv"),
                ("all files", "*.*"),
            ),
            defaultextension=".xml",
        )
        if not filepath:
            return
        self._update_initialdir("save", filepath)

        filepath = Path(filepath).resolve()
        if filepath.suffix in [".csv", ".CSV"]:
            self.parser.save_as_csv(filepath)
        elif filepath.suffix in [".xml", ".musicxml"]:
            self.parser.save_musicxml(filepath)
        elif filepath.suffix in [".pdf", ".PDF"]:
            self.parser.save_as_pdf(filepath)
        else:
            logging.error("unsupport file format %s", filepath.suffix)

    def click_prev(self):
        if not self.parser.is_musicxml_loaded():
            logging.error("please load musicxml first")
            return
        if self.preview_index == 0:
            return
        self.preview_index = max(self.preview_index - 1, 0)

        self.original_image = self.preview_preload[self.preview_index]
        self._resize_image(self.winfo_width(), self.winfo_height())

    def click_next(self):
        if not self.parser.is_musicxml_loaded():
            logging.error("please load musicxml first")
            return
        if self.preview_index == (self.preview_count - 1):
            return
        self.preview_index = min(self.preview_index + 1, self.preview_count - 1)

        self.original_image = self.preview_preload[self.preview_index]
        self._resize_image(self.winfo_width(), self.winfo_height())


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    master = tk.Tk()
    ViolinApp(master).pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    master.mainloop()
