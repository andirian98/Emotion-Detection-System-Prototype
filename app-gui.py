import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, PhotoImage
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from model_train import train_app
from test_cam import test_cam_app
from FacialExpression import main_app, create_folder
from vgg_16 import main_vgg
from ANet import main_alexnet
from GNet import main_googlenet
from data_visualize import visualize
import os

names = set()


class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        global names
        with open("User List.txt", "r") as f:
            x = f.read()
            z = x.strip().split(" ")
            for i in z:
                names.add(i)
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title("Facial Expression Detection")
        self.resizable(False, False)
        self.geometry("720x500")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.active_name = None
        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne, PageTwo, PageThree, PageFour):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            global names
            f = open("User List.txt", "a+")
            for i in names:
                f.write(i + " ")
            self.destroy()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self['background'] = 'white'
        self.controller = controller
        load = Image.open("Untitled-2.png")
        load = load.resize((250, 250), Image.Resampling.LANCZOS) #ANTIALIAS is depreciated
        render = PhotoImage(file='Untitled-2.png')
        img = tk.Label(self, image=render)
        img.image = render
        img.grid(row=0, column=1, rowspan=4, sticky="nsew")

        label = tk.Label(self, text="        Home Page        ", font=self.controller.title_font, fg="#263942")
        label.grid(row=0, sticky="ew")
        button1 = tk.Button(self, text="   New User  ", fg="#ffffff", bg="#4b23db",
                            command=lambda: self.controller.show_frame("PageOne"))
        button2 = tk.Button(self, text="   Old User  ", fg="#ffffff", bg="#e6092a",
                            command=lambda: self.controller.show_frame("PageTwo"))
        button3 = tk.Button(self, text="Quit", fg="#263942", bg="#ffffff", command=self.on_closing)
        button1.grid(row=1, column=0, ipady=3, ipadx=7)
        button2.grid(row=2, column=0, ipady=3, ipadx=2)
        button3.grid(row=3, column=0, ipady=3, ipadx=32)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            global names
            with open("User List.txt", "w") as f:
                for i in names:
                    f.write(i + " ")
            self.controller.destroy()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        load2 = Image.open("Untitled-3.png")
        load2 = load2.resize((250, 250), Image.Resampling.LANCZOS) #ANTIALIAS is depreciated
        render2 = PhotoImage(file='Untitled-3.png')
        img1 = tk.Label(self, image=render2)
        img1.image = render2
        img1.place(x=0, y=0)
        tk.Label(self, text="Enter your name", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, pady=10,
                                                                                            padx=5)
        self.user_name = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.user_name.grid(row=0, column=1, pady=10, padx=10)
        self.buttoncanc = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942",
                                    command=lambda: controller.show_frame("StartPage"))
        self.buttonext = tk.Button(self, text="Next", fg="#ffffff", bg="#263942", command=self.start_training)
        self.buttoncanc.grid(row=1, column=0, pady=10, ipadx=5, ipady=4)
        self.buttonext.grid(row=1, column=1, pady=10, ipadx=5, ipady=4)

    def start_training(self):
        global names
        if self.user_name.get() == "None":
            messagebox.showerror("Error", "Name cannot be 'None'")
            return
        elif self.user_name.get() in names:
            messagebox.showerror("Error", "User already exists!")
            return
        elif len(self.user_name.get()) == 0:
            messagebox.showerror("Error", "Name cannot be empty!")
            return
        name = self.user_name.get()
        names.add(name)
        self.controller.active_name = name
        self.controller.frames["PageTwo"].refresh_names()
        self.controller.show_frame("PageThree")


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        global names
        self.controller = controller
        load4 = Image.open("Untitled-4.png")
        load4 = load4.resize((250, 250), Image.Resampling.LANCZOS)
        render4 = PhotoImage(file='Untitled-4.png')
        img3 = tk.Label(self, image=render4)
        img3.image = render4
        img3.place(x=0, y=0)
        tk.Label(self, text="Select user", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, padx=10,
                                                                                        pady=10)
        self.buttoncanc = tk.Button(self, text="Cancel", command=lambda: controller.show_frame("StartPage"),
                                    bg="#ffffff", fg="#263942")
        self.menuvar = tk.StringVar(self)
        self.dropdown = tk.OptionMenu(self, self.menuvar, *names)
        self.dropdown.config(bg="lightgrey")
        self.dropdown["menu"].config(bg="lightgrey")
        self.buttonext = tk.Button(self, text="Next", command=self.nextfoo, fg="#ffffff", bg="#263942")
        self.dropdown.grid(row=0, column=1, ipadx=8, padx=10, pady=10)
        self.buttoncanc.grid(row=1, ipadx=5, ipady=4, column=0, pady=10)
        self.buttonext.grid(row=1, ipadx=5, ipady=4, column=1, pady=10)

    def nextfoo(self):
        if self.menuvar.get() == "None":
            messagebox.showerror("ERROR", "Name cannot be 'None'")
            return
        self.controller.active_name = self.menuvar.get()
        self.controller.show_frame("PageThree")

    def refresh_names(self):
        global names
        self.menuvar.set('')
        self.dropdown['menu'].delete(0, 'end')
        for name in names:
            self.dropdown['menu'].add_command(label=name, command=tk._setit(self.menuvar, name))


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        load3 = Image.open("Untitled-1.png")
        load3 = load3.resize((250, 250), Image.Resampling.LANCZOS)
        render3 = PhotoImage(file='Untitled-1.png')
        img2 = tk.Label(self, image=render3)
        img2.image = render3
        img2.place(x=0, y=0)
        label = tk.Label(self, text="Menu", font='Helvetica 16 bold')
        label.grid(row=0, column=0, sticky="ew")
        tk.Label(self, text="Enter batch size", fg="#263942", font='Helvetica 12 bold').grid(row=1, column=0, pady=10,
                                                                                             padx=10)
        self.batch_size = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.batch_size.grid(row=1, column=1, pady=5, padx=3)
        tk.Label(self, text="Enter epoch size", fg="#263942", font='Helvetica 12 bold').grid(row=2, column=0, pady=10,
                                                                                             padx=10)
        self.epoch_size = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.epoch_size.grid(row=2, column=1, pady=5, padx=3)
        tk.Label(self, text="Enter learning rate", fg="#263942", font='Helvetica 12 bold').grid(row=3, column=0,
                                                                                                pady=10,
                                                                                                padx=10)
        self.learning_rate = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.learning_rate.grid(row=3, column=1, pady=5, padx=3)
        button1 = tk.Button(self, text="Model Training", fg="#ffffff", bg="#9900ff", command=self.training_model)
        button2 = tk.Button(self, text="Webcam Testing", fg="#ffffff", bg="#6e6664", command=self.testing_camera)
        button3 = tk.Button(self, text="Emotion Detection", fg="#ffffff", bg="#ff003c", command=self.detect_emotion)
        button6 = tk.Button(self, text="Emotion Visualization", fg="#ffffff", bg="#260ef1",
                            command=self.emotion_visualization)
        button4 = tk.Button(self, text="Home Page", command=lambda: self.controller.show_frame("StartPage"),
                            bg="#ffffff", fg="#263942")
        button5 = tk.Button(self, text="Next Page", command=lambda: self.controller.show_frame("PageFour"),
                            bg="#ffffff", fg="#263942")
        button1.grid(row=4, column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button2.grid(row=5, column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button3.grid(row=6, column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button4.grid(row=8, column=1, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button5.grid(row=6, column=1, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button6.grid(row=8, column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)

    def emotion_visualization(self):
        filename = askopenfilename()
        visualize(filename)

    def detect_emotion(self):
        path = str(self.controller.active_name) + '/data'
        if os.path.exists(path):
            messagebox.showerror("Error", "Folder already exists!")
        else:
            create_folder(self.controller.active_name)
            main_app(self.controller.active_name)

    def training_model(self):
        size_batch = self.batch_size.get()
        size_batch = int(size_batch)
        size_epoch = self.epoch_size.get()
        size_epoch = int(size_epoch)
        rate_learn = self.learning_rate.get()
        rate_learn = float(rate_learn)
        folderPath = tk.StringVar()
        folder_selected = filedialog.askdirectory()
        folderPath.set(folder_selected)
        folder = folderPath.get()
        filename = askopenfilename()
        train_app(folder, filename, size_batch, size_epoch, rate_learn)
        # messagebox.showinfo("SUCCESS", "The model has been successfully trained!")

    def testing_camera(self):
        test_cam_app()


class PageFour(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        load3 = Image.open("Untitled-1.png")
        load3 = load3.resize((250, 250), Image.Resampling.LANCZOS)
        render3 = PhotoImage(file='Untitled-1.png')
        img2 = tk.Label(self, image=render3)
        img2.image = render3
        img2.place(x=0, y=0)
        label = tk.Label(self, text="CNN Model Option", font='Helvetica 16 bold')
        label.grid(row=0, column=0, sticky="ew")
        tk.Label(self, text="Enter batch size", fg="#263942", font='Helvetica 12 bold').grid(row=1, column=0, pady=10,
                                                                                             padx=10)
        self.batch_size = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.batch_size.grid(row=1, column=1, pady=5, padx=3)
        tk.Label(self, text="Enter epoch size", fg="#263942", font='Helvetica 12 bold').grid(row=2, column=0, pady=10,
                                                                                             padx=10)
        self.epoch_size = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.epoch_size.grid(row=2, column=1, pady=5, padx=3)
        tk.Label(self, text="Enter learning rate", fg="#263942", font='Helvetica 12 bold').grid(row=3, column=0,
                                                                                                pady=10,
                                                                                                padx=10)
        self.learning_rate = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.learning_rate.grid(row=3, column=1, pady=5, padx=3)
        button4 = tk.Button(self, text="Home Page", command=lambda: self.controller.show_frame("StartPage"),
                            bg="#ffffff", fg="#263942")
        button5 = tk.Button(self, text="Previous Page", command=lambda: self.controller.show_frame("PageThree"),
                            bg="#ffffff", fg="#263942")
        button4.grid(row=5, column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button5.grid(row=6, column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button6 = tk.Button(self, text="VGG-16", fg="#ffffff", bg="#09e677", command=self.doVGG)
        button7 = tk.Button(self, text="AlexNet", fg="#ffffff", bg="#09e677", command=self.doAlexNet)
        button8 = tk.Button(self, text="GoogleNet", fg="#ffffff", bg="#09e677", command=self.doGoogleNet)
        button6.grid(row=4, column=0, sticky="ew", ipadx=5, ipady=4, padx=15, pady=10)
        button7.grid(row=4, column=1, sticky="ew", ipadx=5, ipady=4, padx=15, pady=10)
        button8.grid(row=4, column=2, sticky="ew", ipadx=5, ipady=4, padx=15, pady=10)

    def doVGG(self):
        size_batch = self.batch_size.get()
        size_batch = int(size_batch)
        size_epoch = self.epoch_size.get()
        size_epoch = int(size_epoch)
        rate_learn = self.learning_rate.get()
        rate_learn = float(rate_learn)
        folderPath = tk.StringVar()
        folder_selected = filedialog.askdirectory()
        folderPath.set(folder_selected)
        folder = folderPath.get()
        filename = askopenfilename()
        main_vgg(folder, filename, size_batch, size_epoch, rate_learn)
        # messagebox.showinfo("SUCCESS", "The model has been successfully trained!")

    def doAlexNet(self):
        size_batch = self.batch_size.get()
        size_batch = int(size_batch)
        size_epoch = self.epoch_size.get()
        size_epoch = int(size_epoch)
        rate_learn = self.learning_rate.get()
        rate_learn = float(rate_learn)
        folderPath = tk.StringVar()
        folder_selected = filedialog.askdirectory()
        folderPath.set(folder_selected)

        folder = folderPath.get()
        filename = askopenfilename()
        main_alexnet(folder, filename, size_batch, size_epoch, rate_learn)
        # messagebox.showinfo("SUCCESS", "The model has been successfully trained!")

    def doGoogleNet(self):
        size_batch = self.batch_size.get()
        size_batch = int(size_batch)
        size_epoch = self.epoch_size.get()
        size_epoch = int(size_epoch)
        rate_learn = self.learning_rate.get()
        rate_learn = float(rate_learn)
        folderPath = tk.StringVar()
        folder_selected = filedialog.askdirectory()
        folderPath.set(folder_selected)
        folder = folderPath.get()
        filename = askopenfilename()
        main_googlenet(folder, filename, size_batch, size_epoch, rate_learn)
        # messagebox.showinfo("SUCCESS", "The model has been successfully trained!")


app = MainUI()
# icon1 = ImageTk.PhotoImage(file = 'C:/Users/User/Documents/UiTM/FSKM/CS259/Sem 6 Sub/CSP 650/CSP 650 Prototype/Facial Expression Detection System Prototype/Emotion Detector System/icon-2.ico')
root = tk.Tk()
image = Image.open(
    r"F:\My Documents\UiTM\FSKM\CS259\Sem 6 Sub\CSP 650\CSP 650 Prototype\Facial Expression Detection System Prototype\Emotion Detector System\icon-2.ico")
icon1 = ImageTk.PhotoImage(image)
app.iconphoto(False, icon1)
app.mainloop()
