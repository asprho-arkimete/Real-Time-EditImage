import tkinter as tk
from tkinter import HORIZONTAL, filedialog
from PIL import Image, ImageTk
from tkinterdnd2 import TkinterDnD, DND_FILES
from transformers.video_utils import load_video

import os

def crea_cartelle_utili():
    """Crea le cartelle necessarie se non esistono già."""
    cartelle = ['clips', 'lora', 'lora_ltx', 'model', 'pose']
    for cartella in cartelle:
        os.makedirs(os.path.join(".", cartella), exist_ok=True)

crea_cartelle_utili()
        

# TkinterDnD.Tk() al posto di tk.Tk()
window = TkinterDnD.Tk()
window.title("RealTime_Generation")
window.geometry("1484x910")
window.resizable(False, False)

frame_canvas = tk.Frame(window)
frame_canvas.grid(row=0, column=0, sticky='nw')

frame1 = tk.Canvas(frame_canvas, width=512, height=512, bg='red')
frame1.grid(row=0, column=0, sticky='nw')

frame_input = tk.Frame(frame_canvas)
frame_input.grid(row=0, column=1, sticky='nw')

CANVAS_SIZE = 80

canvas_params = [
    {'bg': 'pink',        'row': 0, 'col': 0, 'text': "reference 1"},
    {'bg': 'light blue',  'row': 0, 'col': 1, 'text': "reference 2"},
    {'bg': 'light green', 'row': 1, 'col': 0, 'text': "reference 3"},
    {'bg': 'orange',      'row': 1, 'col': 1, 'text': "reference 4"},
]

# Percorsi esterni: path_images[0] -> reference 1, ecc.
path_image1,path_image2,path_image3,path_image4 = '', '', '', ''
# Riferimenti alle PhotoImage (altrimenti il garbage collector le cancella)
photo_refs = [None, None, None, None]

photo_canvases = []

VALID_EXT = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')


def set_image(index, canvas, file_path):
    """Carica, ridimensiona e disegna l'immagine sul canvas indicato."""
    global path_image1, path_image2, path_image3, path_image4

    if not file_path.lower().endswith(VALID_EXT):
        print(f"Formato non supportato: {file_path}")
        return None

    try:
        img = Image.open(file_path).convert("RGBA")
    except Exception as e:
        print(f"Impossibile aprire l'immagine: {e}")
        return None

    # Ridimensiona mantenendo le proporzioni dentro 80x80
    w, h = img.size
    scale = min(CANVAS_SIZE / w, CANVAS_SIZE / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    photo = ImageTk.PhotoImage(img)
    canvas.delete("all")
    canvas.create_image(CANVAS_SIZE // 2, CANVAS_SIZE // 2, image=photo)
    photo_refs[index] = photo

    if index == 0:
        path_image1 = file_path
    elif index == 1:
        path_image2 = file_path
    elif index == 2:
        path_image3 = file_path
    elif index == 3:
        path_image4 = file_path

    print(f"Reference {index + 1}: {file_path}")
    return img

def on_click(index, canvas, event=None):
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp")]
    )
    if file_path:
        set_image(index, canvas, file_path)


def on_drop(index, canvas, event):
    # event.data può contenere più file e graffe se il percorso ha spazi
    files = window.tk.splitlist(event.data)
    if files:
        set_image(index, canvas, files[0])   # usa solo il primo file


for i, params in enumerate(canvas_params):
    canvas = tk.Canvas(frame_input, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=params['bg'])
    canvas.grid(row=params['row'], column=params['col'], sticky='nw')
    canvas.create_text((40, 40), text=params['text'], font=('Arial', 10), fill='white')

    # Click -> dialogo file
    canvas.bind('<Button-1>', lambda e, idx=i, c=canvas: on_click(idx, c, e))

    # Drag & drop -> file trascinato da Esplora risorse / Finder
    canvas.drop_target_register(DND_FILES)
    canvas.dnd_bind('<<Drop>>', lambda e, idx=i, c=canvas: on_drop(idx, c, e))

    photo_canvases.append(canvas)

frame_button=tk.Frame(frame_canvas)
frame_button.grid(row=0,column=2,sticky='nw')

lab_steps = tk.Label(frame_button, text="STEPS")
lab_steps.grid(row=0, column=0, sticky='nw')
steps = tk.Scale(frame_button, from_=1, to=50, orient=tk.HORIZONTAL)
steps.grid(row=1, column=0, sticky='nw',padx=5)
steps.set(2)

lab_time = tk.Label(frame_button, text="Attesa Generazione (s)")
lab_time.grid(row=0, column=1, sticky='nw')

attesa = tk.Scale(frame_button, from_=1, to=10, orient=tk.HORIZONTAL)
attesa.grid(row=1, column=1, sticky='nw', padx=5)
attesa.set(2)

count_time = 0
Generazione_TimeReal = False
timer_id = None   # id del timer di after(), serve per poterlo annullare
testo_modificato = False   # True = c'è del testo nuovo da generare

import gc
import os
import torch
import tkinter as tk
from PIL import Image, ImageTk
from diffusers import Flux2KleinPipeline
from optimum.quanto import freeze, qfloat8, quantize

old_lora1 = ''
old_lora2 = ''
pipe = None          # il modello resta in memoria finché le LoRA non cambiano
result_photo = None  # riferimento, altrimenti Tkinter cancella l'immagine
FINALLY=False


def f_resize(path, size=128):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    rw, rh = size, size
    if w >= h:
        rh = max(16, (rw * h) // w)
    else:
        rw = max(16, (rh * w) // h)
    rw, rh = max(16, rw // 16 * 16), max(16, rh // 16 * 16)   # multipli di 16
    return img.resize((rw, rh), Image.Resampling.BICUBIC)

import queue
import threading as T

coda = queue.Queue()      # il thread di generazione mette qui i risultati
generando = False         # True mentre una generazione è in corso
result_image = None       # ultima immagine PIL generata (per salvarla)

import torch
from transformers import MarianMTModel, MarianTokenizer

_MT_NAME = "Helsinki-NLP/opus-mt-it-en"
_mt_tok = None
_mt_model = None


def traduci_ita_eng(testo):
    global _mt_tok, _mt_model
    if _mt_model is None:
        _mt_tok = MarianTokenizer.from_pretrained(_MT_NAME)
        _mt_model = MarianMTModel.from_pretrained(_MT_NAME).eval()

    tokens = _mt_tok(testo, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = _mt_model.generate(**tokens, max_length=256, num_beams=4)
    return _mt_tok.decode(out[0], skip_special_tokens=True)

from PIL import Image, ImageOps

def snap(v, m=32):
    """Multiplo di m più vicino (minimo m)."""
    return max(m, int(round(v / m)) * m)


def calc_size(img, size, m=32):
    """Lato maggiore = size, proporzione mantenuta, multipli di m."""
    w, h = img.size
    if w >= h:
        return snap(size, m), snap(size * h / w, m)
    return snap(size * w / h, m), snap(size, m)


def carica_originale(path):
    """Apre l'immagine applicando la rotazione EXIF."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def resize_ratio(img, wg, hg):
    """Scala senza deformare e ritaglia al centro solo l'eccesso (pochi pixel)."""
    w, h = img.size
    scale = max(wg / w, hg / h)
    nw, nh = round(w * scale), round(h * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (nw - wg) // 2, (nh - hg) // 2
    return img.crop((left, top, left + wg, top + hg))


def trova_lora(value):
    """Restituisce il path del file lora (con estensione valida) o None."""
    estensioni = ('.safetensors', '.pt', '.ckpt')
    if not value or value == 'nolora':
        return None
    # se ha già un'estensione valida, usa il nome così com'è
    if value.lower().endswith(estensioni):
        p = os.path.join("lora", value)
        return p if os.path.exists(p) else None
    # altrimenti prova ad aggiungere le estensioni supportate
    for ext in estensioni:
        p = os.path.join("lora", value + ext)
        if os.path.exists(p):
            return p
    return None


active_names = []   # globale: adapter caricati nella pipe corrente
def flux2_RealTime(prompt, n_steps, refs, new1, new2, size, final=False):
    """Gira nel thread secondario: NON tocca nessun widget. Restituisce l'immagine PIL."""
    global pipe, old_lora1, old_lora2, active_names, paths_loras, lora1, lora2, scale_lora1, scale_lora2

    device = "cuda"
    dtype = torch.bfloat16

    idx = lora1.current()
    new1 = paths_loras[idx] if 0 <= idx < len(paths_loras) else lora1.get()
    new2 = lora2.get()

    pesi = {"lora1": scale_lora1.get(), "lora2": scale_lora2.get()}

    # Ricarica la pipe solo se è la prima volta o è cambiata una LoRA (non i pesi)
    if pipe is None or new1 != old_lora1 or new2 != old_lora2:
        if pipe is not None:
            del pipe
            pipe = None
            gc.collect()
            torch.cuda.empty_cache()

        pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B", dtype=dtype)

        active_names = []
        for name, value in (("lora1", new1), ("lora2", new2)):
            path_lora = trova_lora(value)
            if path_lora:
                pipe.load_lora_weights(path_lora, adapter_name=name)
                active_names.append(name)
            elif value and value != 'nolora':
                print(f"LoRA non trovata o estensione non valida: {value}")

        # NIENTE fuse_lora() / unload_lora_weights()

        quantize(pipe.transformer, weights=qfloat8)
        freeze(pipe.transformer)
        quantize(pipe.text_encoder, weights=qfloat8)
        freeze(pipe.text_encoder)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        old_lora1, old_lora2 = new1, new2

    # A ogni chiamata: applica solo i pesi correnti, senza ricaricare
    if active_names:
        pipe.set_adapters(active_names, adapter_weights=[pesi[n] for n in active_names])

    # --- Riferimenti: proporzione calcolata sugli ORIGINALI, mai deformati ---
    originali = [carica_originale(p) for p in refs]

    images = []
    for img in originali:
        rw, rh = calc_size(img, size)
        images.append(resize_ratio(img, rw, rh))

    if not final and n_steps > 2:
        print("Real time: steps limitati a 2")
        n_steps = 2

    # --- Dimensione di output ---
    wg, hg = size, size
    if images and "clothesonoffv2" in new1:
        # stessa dimensione (e proporzione) del primo riferimento
        wg, hg = images[0].size

    print(f"Generazione {wg}x{hg}, steps={n_steps}, prompt='{prompt[:30]}...'")

    return pipe(
        prompt=prompt,
        image=images if images else None,
        height=hg,
        width=wg,
        guidance_scale=1.0,
        num_inference_steps=n_steps,
        generator=torch.Generator(device=device).manual_seed(0)
    ).images[0]


def worker(prompt, n_steps, refs, new1, new2, size, final):
    try:
        image = flux2_RealTime(prompt, n_steps, refs, new1, new2, size, final)
        coda.put(("ok", image, final))
    except Exception as e:
        coda.put(("err", str(e), final))


from deep_translator import GoogleTranslator
def avvia_generazione(size, final=False):
    """Legge i widget (thread principale) e avvia il thread. False se già occupato."""
    global generando
    if generando:
        return False
    generando = True

    prompt_ita = text.get("1.0", "end-1c")
    try:
        prompt = traduci_ita_eng(prompt_ita)
    except Exception as e:
        print(f"Errore nella traduzione automatica: {e}")
        prompt = prompt_ita

    n_steps = int(steps.get())
    new1 = lora1.get().strip()
    new2 = lora2.get().strip()
    new1 = '' if new1 == 'nolora' else new1
    new2 = '' if new2 == 'nolora' else new2
    refs = [p for p in (path_image1, path_image2, path_image3, path_image4)
            if p and os.path.exists(p)]

    T.Thread(target=worker,
             args=(prompt, n_steps, refs, new1, new2, size, final),
             daemon=True).start()
    return True

os.makedirs("clips",exist_ok=True)
def controlla_coda():
    """Gira nel thread principale ogni 100 ms: riceve i risultati e aggiorna la GUI."""
    global generando, result_photo, result_image
    try:
        while True:
            tipo, dato, final = coda.get_nowait()
            generando = False
            if tipo == "ok":
                result_image = dato

                # anteprima: lato maggiore = 512, proporzione mantenuta
                w, h = dato.size
                if w >= h:
                    rw, rh = 512, max(1, (512 * h) // w)
                else:
                    rw, rh = max(1, (512 * w) // h), 512

                shown = dato if dato.size == (rw, rh) else dato.resize(
                    (rw, rh), Image.Resampling.BICUBIC)
                result_photo = ImageTk.PhotoImage(shown)
                frame1.delete("all")
                frame1.create_image(0, 0, anchor='nw', image=result_photo)

                if final:
                    path = percorso_libero("./clips//flux-klein_finally.png")
                    dato.save(path)
                    lab_text_change.config(text=f"SALVATO: {path}", fg='blue')
                else:
                    lab_text_change.config(text="GENERATO - in attesa di modifiche", fg='blue')
            else:
                print(f"Errore di generazione: {dato}")
                lab_text_change.config(text="ERRORE (vedi console)", fg='red')
    except queue.Empty:
        pass
    window.after(100, controlla_coda)


def tick():
    """Viene chiamata ogni secondo finché RealTime è attivo."""
    global count_time, timer_id, testo_modificato

    if not Generazione_TimeReal:
        return

    if testo_modificato:
        count_time += 1
        print(f"Time: {count_time}")

        if count_time >= int(attesa.get()):
            if generando:
                # una generazione è ancora in corso: aspetta, testo_modificato resta True
                lab_text_change.config(text="Attendo fine generazione...", fg='orange')
            else:
                lab_text_change.config(text="GENERAZIONE IMMAGINE", fg='green')
                count_time = 0
                testo_modificato = False
                avvia_generazione(512)
        else:
            lab_text_change.config(text=f"MODIFICA TESTO, Time:{count_time}", fg='red')

    timer_id = window.after(1000, tick)


def f_gen_REALt():
    global count_time, Generazione_TimeReal, timer_id, testo_modificato

    count_time = 0

    if not Generazione_TimeReal:
        Generazione_TimeReal = True
        testo_modificato = True            # al primo avvio genera dal prompt attuale
        gen_realtime.config(bg='light green')
        timer_id = window.after(1000, tick)
    else:
        Generazione_TimeReal = False
        gen_realtime.config(bg='light blue')
        lab_text_change.config(text="RealTime fermo", fg='gray')
        if timer_id is not None:
            window.after_cancel(timer_id)
            timer_id = None

gen_realtime = tk.Button(frame_button, text="RealTime", bg='light blue', command=f_gen_REALt)
gen_realtime.grid(row=2, column=1, sticky='nw', padx=5)


def f_bozza():
    if result_image is not None:
        path = percorso_libero("./clips//flux-klein_bozza.png")
        result_image.save(path)
        print(f"Bozza salvata: {path}")

Salva_bozza = tk.Button(frame_button, text="Salva Bozza",command= f_bozza)
Salva_bozza.grid(row=0, column=2, sticky='nw',padx=5)

def f_finaly():
    if avvia_generazione(1024, final=True):
        lab_text_change.config(text="GENERAZIONE FINALE 1024...", fg='green')
    else:
        lab_text_change.config(text="Generazione in corso, riprova tra poco", fg='orange')

Salva_final = tk.Button(frame_button, text="Salva Generazione", bg='light green',command=f_finaly)
Salva_final.grid(row=0, column=3, sticky='nw',padx=5)

import tkinter as tk
from tkinter import ttk
import os
import json
paths_loras=[]
prompt_azioni=[]
def load_lora(event=None):
    global paths_loras,prompt_azioni
    lora_dir = 'lora'
    azioni=[]
    paths_loras=[]
    prompt_azioni=[]
    with open("presets_prompts.json", encoding="utf-8") as f:
        presets = json.load(f)
    for p in presets:
        azioni.append(p['azione'])
        prompt_azioni.append(p['prompt'])
        paths_loras.append(p['nome_lora1'])
    if os.path.isdir(lora_dir):
        for l in os.listdir(lora_dir):
            if not l in paths_loras:
                azioni.append(l)
                prompt_azioni.append('')
                paths_loras.append(l)
    lora1['values'] = azioni
    
    if not os.path.isdir(lora_dir):
        models = ['nolora']
    else:
        models = ['nolora'] + [os.path.splitext(os.path.basename(m))[0]
                               for m in os.listdir(lora_dir)
                               if m.lower().endswith(('.pt', '.ckpt', '.safetensors'))]
    lora2['values'] = models

 
    
    

lab_lora1 = tk.Label(frame_button, text='Lora 1')
lab_lora1.grid(row=1, column=2)
lora1 = ttk.Combobox(frame_button, values=[], postcommand=load_lora)
lora1.grid(row=2, column=2)
lora1.set('nolora')

scale_lora1 = tk.Scale(frame_button, from_=0.10, to=2.00, resolution=0.10,orient='horizontal', length=100)
scale_lora1.set(0.8)
scale_lora1.grid(row=3, column=2)
 



lab_lora2 = tk.Label(frame_button, text='Lora 2')
lab_lora2.grid(row=1, column=3)
lora2 = ttk.Combobox(frame_button, values=[], postcommand=load_lora)
lora2.grid(row=2, column=3)
lora2.set('nolora')

scale_lora2 = tk.Scale(frame_button, from_=0.10, to=2.00, resolution=0.10,
                       orient='horizontal', length=100)
scale_lora2.set(0.8)
scale_lora2.grid(row=3, column=3)


def f_cambio(event=None):
    global testo_modificato, count_time
    testo_modificato = True
    count_time = 0
    global text, prompt_azioni, paths_loras, lora1
    idx = lora1.current()
    if idx < 0 or idx >= len(prompt_azioni):
        return
    try:
        prompt = prompt_azioni[idx]
        path = paths_loras[idx]
    except IndexError:
        print("Indice Lora fuori dai limiti.")
        return
    print(f"prompt: {prompt}")
    print(f"paths: {path}")
    text.delete('1.0', 'end')
    text.insert('1.0', prompt)


lora1.bind('<<ComboboxSelected>>', f_cambio)
lora2.bind('<<ComboboxSelected>>', f_cambio)



frame_text=tk.Frame(window,bg='light gray')
frame_text.grid(row=3,column=0,padx=5,pady=5,sticky='nw')

lab_prompt=tk.Label(frame_text,text="PROMPT")
lab_prompt.grid(row=0,column=0)

text=tk.Text(frame_text,width=80,height=6)
text.grid(row=1, column=0, padx=5, pady=5, sticky='nw')

def f_reset(event=None):
    """Ogni modifica del testo azzera il timer e segna che c'è da generare."""
    global count_time, testo_modificato
    if text.edit_modified():
        count_time = 0
        testo_modificato = True
        if Generazione_TimeReal:
            lab_text_change.config(text="MODIFICA TESTO, Time:0", fg='red')
        text.edit_modified(False)          # riarma l'evento <<Modified>>

text.bind('<<Modified>>', f_reset)


lab_text_change = tk.Label(frame_text, text="MODIFICA TESTO", font=('Arial', 10), fg='red')
lab_text_change.grid(row=2, column=0, padx=5, pady=5, sticky='n')  # Cambiato sticky in 'n' per centrare sotto la text box

import os
from moviepy import VideoFileClip   # moviepy 2.x (con la 1.x: from moviepy.editor import VideoFileClip)
# (da avere già in alto: tk, Image, ImageTk, DND_FILES, window = TkinterDnD.Tk())

# ------------------------------------------------------------------
# VARIABILI GLOBALI
# ------------------------------------------------------------------
path_primo_fotogramma = ''
path_ultimo_fotogramma = ''
indice_primo = None
indice_ultimo = None
selezione = ''

paths_image_canvas = []     # un percorso per ogni fotogramma (None se vuoto)
raccogli_canvas = []
info_clip = {}              # cache: path -> (fps, n_frame, durata)

COLORE_NORMALE = 'black'
COLORE_PRIMO = 'green'
COLORE_ULTIMO = 'blue'

ESTENSIONI_IMMAGINI = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
ESTENSIONI_VIDEO = ('.mp4', '.mkv', '.mov', '.avi', '.webm')

dir_clip = "clips"
os.makedirs(dir_clip, exist_ok=True)


# ------------------------------------------------------------------
# FUNZIONI DI SUPPORTO
# ------------------------------------------------------------------
def e_video(path):
    return path is not None and path.lower().endswith(ESTENSIONI_VIDEO)


def info_video(path):
    """Restituisce (fps, n_frame, durata) della clip, con cache."""
    if path not in info_clip:
        with VideoFileClip(path) as v:
            info_clip[path] = (v.fps, int(v.duration * v.fps), v.duration)
    return info_clip[path]


def percorso_libero(percorso):
    """Se il file esiste già, aggiunge _1, _2, ... per non sovrascriverlo."""
    base, ext = os.path.splitext(percorso)
    n = 1
    while os.path.exists(percorso):
        percorso = f"{base}_{n}{ext}"
        n += 1
    return percorso


def ridimensiona(img):
    """Ridimensiona l'immagine a massimo 90x90 mantenendo le proporzioni."""
    w, h = img.size
    if w >= h:
        rw, rh = 90, max(1, (90 * h) // w)
    else:
        rw, rh = max(1, (90 * w) // h), 90
    return img.resize((rw, rh), Image.Resampling.BICUBIC)


def aggiorna_testo_elimina():
    """Aggiorna il testo del bottone Elimina in base alla selezione corrente."""
    if selezione == '' or selezione >= len(raccogli_canvas):
        btn_Elimina.config(text='Elimina')
    else:
        path = paths_image_canvas[selezione]
        nome = os.path.basename(path) if path else f'frame {selezione} (vuoto)'
        btn_Elimina.config(text=f'Elimina {nome}')


def aggiorna_percorsi_selezione():
    """Riallinea path_primo/ultimo e il testo di Elimina alle liste attuali."""
    global path_primo_fotogramma, path_ultimo_fotogramma
    path_primo_fotogramma = (paths_image_canvas[indice_primo] or '') if indice_primo is not None else ''
    path_ultimo_fotogramma = (paths_image_canvas[indice_ultimo] or '') if indice_ultimo is not None else ''
    aggiorna_testo_elimina()


def aggiorna_slider(path):
    """Imposta il massimo dello slider all'ultimo frame valido della clip (o lo disabilita)."""
    scorri_frame.config(command='')          # evita che set() ridisegni il fotogramma
    try:
        scorri_frame.config(state='normal')
        if e_video(path):
            fps, n_frame, durata = info_video(path)
            scorri_frame.config(to=max(1, n_frame - 1))   # ultimo frame valido = n_frame - 1
            scorri_frame.set(0)
        else:
            scorri_frame.config(to=1)
            scorri_frame.set(0)
            scorri_frame.config(state='disabled')
    except Exception as e:
        print(f"Impossibile leggere le informazioni della clip: {e}")
        scorri_frame.config(to=1)
        scorri_frame.set(0)
        scorri_frame.config(state='disabled')
    finally:
        scorri_frame.config(command=f_scorri_fotogramma)


# ------------------------------------------------------------------
# MOSTRA IMMAGINE / CLIP NEL CANVAS
# ------------------------------------------------------------------
def mostra_immagine(canvas, indice, path):
    """Mostra immagine o frame di un video nel canvas.
    L'indice reale viene ricavato dal canvas, quindi 'indice' può essere obsoleto."""
    if canvas not in raccogli_canvas:
        return False

    estensione = path.lower()

    try:
        if estensione.endswith(ESTENSIONI_IMMAGINI):
            img = Image.open(path)

        elif estensione.endswith(ESTENSIONI_VIDEO):
            with VideoFileClip(path) as video:
                t = min(1.0, video.duration / 2)       # evita il primo frame, spesso nero
                frame = video.get_frame(t)             # array numpy RGB
            img = Image.fromarray(frame)

        else:
            print("File non supportato")
            return False

    except Exception as e:
        print(f"Impossibile leggere il file: {e}")
        return False

    foto = ImageTk.PhotoImage(ridimensiona(img))
    canvas.delete("all")
    canvas.create_image(45, 45, image=foto)

    # indicatore "▶" per distinguere le clip video dalle immagini
    if e_video(path):
        canvas.create_text(8, 8, text="▶", fill="white", font=('Arial', 10, 'bold'))

    canvas.image = foto                   # mantiene il riferimento all'immagine

    idx = raccogli_canvas.index(canvas)
    paths_image_canvas[idx] = path

    aggiorna_percorsi_selezione()
    if idx == selezione:                  # se è il fotogramma selezionato, riallinea lo slider
        aggiorna_slider(path)
    return True


# timeline video: nuovo nome
def on_drop_timeline(event, canvas):
    files = window.tk.splitlist(event.data)
    if files and canvas in raccogli_canvas:
        mostra_immagine(canvas, raccogli_canvas.index(canvas), files[0])


# ------------------------------------------------------------------
# PANNELLO SCROLLABILE
# ------------------------------------------------------------------
frame_timeline = tk.Frame(window, bg='light gray')
frame_timeline.grid(row=4, column=0, sticky='nsew')

contenitore = tk.Canvas(frame_timeline, bg='gray', width=400, height=120,
                        highlightthickness=0)
scroll_y = tk.Scrollbar(frame_timeline, orient='vertical', command=contenitore.yview)
scroll_x = tk.Scrollbar(frame_timeline, orient='horizontal', command=contenitore.xview)
contenitore.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

contenitore.grid(row=0, column=0, sticky='nsew')
scroll_y.grid(row=0, column=1, sticky='ns')
scroll_x.grid(row=1, column=0, sticky='ew')

frame_timeline.rowconfigure(0, weight=1)
frame_timeline.columnconfigure(0, weight=1)

frame_interno = tk.Frame(contenitore, bg='gray')
contenitore.create_window((0, 0), window=frame_interno, anchor='nw')


def aggiorna_scrollregion(event=None):
    contenitore.configure(scrollregion=contenitore.bbox('all'))

frame_interno.bind('<Configure>', aggiorna_scrollregion)


# ------------------------------------------------------------------
# SELEZIONE PRIMO / ULTIMO FOTOGRAMMA
# ------------------------------------------------------------------
def aggiorna_bordi():
    """Ridisegna i bordi in base agli indici selezionati."""
    for k, cv in enumerate(raccogli_canvas):
        if k == indice_primo:
            cv.config(highlightbackground=COLORE_PRIMO, highlightthickness=3)
        elif k == indice_ultimo:
            cv.config(highlightbackground=COLORE_ULTIMO, highlightthickness=3)
        else:
            cv.config(highlightbackground=COLORE_NORMALE, highlightthickness=1)


def f_seleziona(canvas):
    global indice_primo, indice_ultimo, selezione

    if canvas not in raccogli_canvas:
        return
    i = raccogli_canvas.index(canvas)     # indice sempre aggiornato

    if i == indice_primo:                 # clic sul primo: lo deseleziona
        indice_primo = None
    elif i == indice_ultimo:              # clic sull'ultimo: lo deseleziona
        indice_ultimo = None
    elif indice_primo is None:            # primo clic: imposta il primo
        indice_primo = i
    else:                                 # altrimenti imposta/sostituisce l'ultimo
        indice_ultimo = i

    # se l'ultimo precede il primo, li scambia
    if indice_primo is not None and indice_ultimo is not None \
            and indice_ultimo < indice_primo:
        indice_primo, indice_ultimo = indice_ultimo, indice_primo

    # se il fotogramma cliccato è stato deselezionato, non resta "selezionato"
    selezione = i if i in (indice_primo, indice_ultimo) else ''

    aggiorna_bordi()
    aggiorna_percorsi_selezione()         # aggiorna path_primo/ultimo e testo di Elimina

    # slider: massimo = numero di frame della clip selezionata
    aggiorna_slider(paths_image_canvas[selezione] if selezione != '' else None)

    print(f"primo fotogramma: {indice_primo} -> {path_primo_fotogramma}; "
          f"ultimo fotogramma: {indice_ultimo} -> {path_ultimo_fotogramma}")


# ------------------------------------------------------------------
# AGGIUNTA CANVAS IN ORIZZONTALE
# ------------------------------------------------------------------
def aggiungi_canvas():
    indice = len(raccogli_canvas)

    c = tk.Canvas(frame_interno, bg='#ff596c', width=90, height=90,
                  highlightthickness=1, highlightbackground=COLORE_NORMALE)
    c.create_text(45, 45, text=f'frame:{indice}\nclip:{indice}',
                  font=('Arial', 10), fill='black')
    c.grid(row=0, column=indice, padx=2, pady=2)

    # trascina un'immagine/clip sul fotogramma (si passa il canvas, non l'indice)
    c.drop_target_register(DND_FILES)
    c.dnd_bind('<<Drop>>', lambda event, cv=c: on_drop_timeline(event, cv))

    # clic sinistro per selezionare primo / ultimo fotogramma
    c.bind('<Button-1>', lambda event, cv=c: f_seleziona(cv))

    raccogli_canvas.append(c)
    paths_image_canvas.append(None)   # posto riservato per il percorso

    window.update_idletasks()
    contenitore.xview_moveto(1.0)


# ------------------------------------------------------------------
# SCORRI FRAME (lo slider conta i frame)
# ------------------------------------------------------------------
def f_scorri_fotogramma(valore=None):
    """Mostra nel fotogramma selezionato il frame della clip alla posizione dello slider."""
    if selezione == '' or selezione >= len(raccogli_canvas) \
            or not e_video(paths_image_canvas[selezione]):
        print("Nessuna clip video selezionata")
        return

    path = paths_image_canvas[selezione]
    try:
        fps, n_frame, durata = info_video(path)

        # il frame richiesto non può mai superare l'ultimo frame reale
        frame_richiesto = min(int(scorri_frame.get()), n_frame - 1)
        frame_richiesto = max(0, frame_richiesto)

        # un frame intero di margine prima della fine, per non uscire dal file
        t = frame_richiesto / fps
        t = min(t, max(0, durata - 1.5 / fps))

        with VideoFileClip(path) as video:
            frame = video.get_frame(t)
        img = Image.fromarray(frame)
    except Exception as e:
        print(f"Impossibile leggere il frame: {e}")
        return

    foto = ImageTk.PhotoImage(ridimensiona(img))
    cv = raccogli_canvas[selezione]
    cv.delete('all')
    cv.create_image(45, 45, image=foto)
    cv.create_text(8, 8, text="▶", fill="white", font=('Arial', 10, 'bold'))
    cv.image = foto                      # mantiene il riferimento all'immagine


# ------------------------------------------------------------------
# DIVIDI CLIP
# ------------------------------------------------------------------
def scrivi_clip(clip, out):
    """Scrive la clip su disco: prova l'encoder Nvidia, altrimenti usa la CPU."""
    try:
        clip.write_videofile(out, codec='h264_nvenc', audio_codec='aac', logger=None)
    except Exception:
        clip.write_videofile(out, codec='libx264', audio_codec='aac', logger=None)


def aggiungi_canvas_dopo(indice):
    """Crea un nuovo canvas subito dopo 'indice' e aggiorna gli indici di selezione."""
    global indice_primo, indice_ultimo, selezione

    aggiungi_canvas()                          # lo crea in fondo
    c = raccogli_canvas.pop()
    p = paths_image_canvas.pop()
    raccogli_canvas.insert(indice + 1, c)
    paths_image_canvas.insert(indice + 1, p)

    # i fotogrammi dopo 'indice' scalano di una posizione
    def sposta(idx):
        return idx + 1 if (idx is not None and idx != '' and idx > indice) else idx

    indice_primo = sposta(indice_primo)
    indice_ultimo = sposta(indice_ultimo)
    selezione = sposta(selezione)

    rinumera_canvas()                          # riposiziona i canvas nella griglia
    return c


def f_dividi_clip():
    # la clip da dividere è quella selezionata (la stessa mostrata dallo slider)
    if selezione == '' or selezione >= len(raccogli_canvas):
        print("Seleziona una clip")
        return

    indice = selezione
    path = paths_image_canvas[indice]

    if not e_video(path):
        print("La selezione non è una clip video")
        return

    try:
        fps, n_frame, durata = info_video(path)
    except Exception as e:
        print(f"Impossibile leggere la clip: {e}")
        return

    frame_taglio = int(scorri_frame.get())
    if frame_taglio <= 0 or frame_taglio >= n_frame - 1:
        print("Posiziona lo slider tra il primo e l'ultimo frame per scegliere il taglio")
        return
    t = min(frame_taglio / fps, durata)

    nome, ext = os.path.splitext(os.path.basename(path))      # solo il nome del file
    out1 = percorso_libero(os.path.join(dir_clip, f"{nome}_parte1{ext}"))
    out2 = percorso_libero(os.path.join(dir_clip, f"{nome}_parte2{ext}"))

    try:
        with VideoFileClip(path) as video:
            parte1 = video.subclipped(0, t)                # moviepy 1.x: subclip
            parte2 = video.subclipped(t, video.duration)
            scrivi_clip(parte1, out1)
            scrivi_clip(parte2, out2)
            parte1.close()
            parte2.close()
    except Exception as e:
        print(f"Errore durante la divisione: {e}")
        for f in (out1, out2):                             # pulisce i file incompleti
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
        return

    # parte 1 sostituisce la clip selezionata, parte 2 va subito dopo
    mostra_immagine(raccogli_canvas[indice], indice, out1)
    nuovo_canvas = aggiungi_canvas_dopo(indice)
    mostra_immagine(nuovo_canvas, indice + 1, out2)

    aggiorna_bordi()
    aggiorna_percorsi_selezione()
    aggiorna_slider(paths_image_canvas[selezione] if selezione != '' else None)
    window.update_idletasks()
    aggiorna_scrollregion()
    print(f"Creati: {out1} e {out2}")


# ------------------------------------------------------------------
# ELIMINA
# ------------------------------------------------------------------
def rinumera_canvas():
    """Riposiziona i canvas nella griglia e riscrive i numeri dei fotogrammi vuoti."""
    for k, cv in enumerate(raccogli_canvas):
        cv.grid(row=0, column=k, padx=2, pady=2)
        if paths_image_canvas[k] is None:          # fotogramma vuoto: aggiorna il testo
            cv.delete('all')
            cv.create_text(45, 45, text=f'frame:{k}\nclip:{k}',
                           font=('Arial', 10), fill='black')


def f_elimina():
    global selezione, indice_primo, indice_ultimo

    print(f"Indice selezionato: {selezione}")
    if selezione == '' or selezione >= len(raccogli_canvas):
        print("Seleziona una clip")
        return

    i = selezione
    path_eliminato = paths_image_canvas[i]
    canvas = raccogli_canvas.pop(i)
    paths_image_canvas.pop(i)
    canvas.destroy()                       # rimuove il widget dalla timeline

    # cancella dal disco solo le clip create dalla divisione (cartella 'clips'),
    # e solo se nessun altro fotogramma le usa ancora
    if path_eliminato and path_eliminato not in paths_image_canvas \
            and os.path.dirname(os.path.abspath(path_eliminato)) == os.path.abspath(dir_clip):
        info_clip.pop(path_eliminato, None)
        try:
            os.remove(path_eliminato)
        except OSError as e:
            print(f"Impossibile cancellare il file: {e}")

    # aggiorna gli indici di selezione dopo la rimozione
    def sistema(idx):
        if idx is None or idx == i:
            return None                    # era quello eliminato
        return idx - 1 if idx > i else idx

    indice_primo = sistema(indice_primo)
    indice_ultimo = sistema(indice_ultimo)
    selezione = ''

    rinumera_canvas()
    aggiorna_bordi()
    aggiorna_percorsi_selezione()
    aggiorna_slider(None)
    window.update_idletasks()
    aggiorna_scrollregion()


# ------------------------------------------------------------------
# BOTTONI E SLIDER
# ------------------------------------------------------------------
frame_button_video = tk.Frame(window, bg='light blue')
frame_button_video.grid(row=5, column=0, sticky='wn')

btn = tk.Button(frame_button_video, text='Aggiungi canvas', bg='light green',
                command=aggiungi_canvas)
btn.grid(row=0, column=0, sticky='w', padx=2, pady=2)

lab_scorri = tk.Label(frame_button_video, text='Scorri_Frame')
lab_scorri.grid(row=0, column=1, sticky='wn', padx=2, pady=2)

scorri_frame = tk.Scale(frame_button_video, from_=0, to=1, orient=tk.HORIZONTAL,
                        length=100, command=f_scorri_fotogramma, state='disabled')
scorri_frame.grid(row=1, column=1, sticky='wn', padx=2, pady=2)

btn_dividi = tk.Button(frame_button_video, text='Dividi Clip', bg='burlywood',
                       command=f_dividi_clip)
btn_dividi.grid(row=0, column=2, sticky='w', padx=2, pady=2)

btn_Elimina = tk.Button(frame_button_video, text='Elimina', bg='red',
                        command=f_elimina)
btn_Elimina.grid(row=0, column=3, sticky='w', padx=2, pady=2)

import gc
import os
import re
import threading
import time

import torch
import tkinter as tk
from PIL import Image
from diffusers import (
    BitsAndBytesConfig,
    LTX2ConditionPipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video
from optimum.quanto import freeze, qfloat8, quantize
from transformers import MarianMTModel, MarianTokenizer

try:
    from moviepy.editor import VideoFileClip   # moviepy 1.x
except ImportError:
    from moviepy import VideoFileClip          # moviepy 2.x

ESTENSIONI_IMMAGINI = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
ESTENSIONI_VIDEO = ('.mp4', '.mov', '.avi', '.mkv', '.webm')

pipe_video = None
old_lora_video = None      # None = pipe mai caricata

# ---------------------------------------------------------------------------
# Traduzione IT -> EN (offline, modello caricato una sola volta, resta su CPU)
# Richiede: pip install sentencepiece
# ---------------------------------------------------------------------------
_MT_NAME = "Helsinki-NLP/opus-mt-it-en"
_mt_tok = None
_mt_model = None


import torch
from transformers import MarianMTModel, MarianTokenizer

_MT_NAME = "Helsinki-NLP/opus-mt-it-en"
_mt_tok = None
_mt_model = None


def traduci_ita_eng(testo):
    global _mt_tok, _mt_model
    if _mt_model is None:
        _mt_tok = MarianTokenizer.from_pretrained(_MT_NAME)
        _mt_model = MarianMTModel.from_pretrained(_MT_NAME).eval()

    tokens = _mt_tok(testo, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = _mt_model.generate(**tokens, max_length=256, num_beams=4)
    return _mt_tok.decode(out[0], skip_special_tokens=True)


def traduci_prompt(testo):
    """Traduce frase per frase (Marian rende meglio su frasi brevi)."""
    frasi = [f.strip() for f in re.split(r'(?<=[.!?])\s+', testo) if f.strip()]
    return " ".join(traduci_ita_eng(f) for f in frasi)


# ---------------------------------------------------------------------------
# Immagini di condizione
# ---------------------------------------------------------------------------
def fit_crop(img, w, h, anchor_y=0.0, anchor_x=0.5):
    """Scala l'immagine fino a coprire (w, h) e croppa.
    anchor_y: 0.0 = tiene la parte alta (taglia sotto), 0.5 = centro, 1.0 = tiene la parte bassa.
    anchor_x: 0.0 = tiene la sinistra, 0.5 = centro, 1.0 = tiene la destra.
    """
    iw, ih = img.size
    scale = max(w / iw, h / ih)
    nw, nh = round(iw * scale), round(ih * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = round((nw - w) * anchor_x)
    top = round((nh - h) * anchor_y)
    return img.crop((left, top, left + w, top + h))


def carica_condizione(path, ultimo_frame):
    """Restituisce un'immagine PIL (dimensione originale) o None."""
    if not path or not os.path.exists(path):
        return None
    p = path.lower()
    if p.endswith(ESTENSIONI_IMMAGINI):
        return Image.open(path).convert("RGB")
    elif p.endswith(ESTENSIONI_VIDEO):
        with VideoFileClip(path) as v:
            t = max(0, v.duration - 1.5 / v.fps) if ultimo_frame else 0
            return Image.fromarray(v.get_frame(t)).convert("RGB")
    print(f"Formato sconosciuto: {path}")
    return None


# ---------------------------------------------------------------------------
# Generazione
# Si assume che nel resto dell'app esistano: risoluzione, fps_V, frames_V,
# text (widget tkinter) e la funzione percorso_libero().
# ---------------------------------------------------------------------------
from safetensors.torch import load_file
def carica_lora_convertita(pipe, path, adapter_name):
    sd = load_file(path)
    print("Esempio chiavi LoRA:", list(sd.keys())[:6])
    nuovo = {}
    for k, v in sd.items():
        if k.endswith(".alpha"):
            continue
        nk = k
        if nk.startswith("diffusion_model."):
            nk = "transformer." + nk[len("diffusion_model."):]
        nk = nk.replace(".lora_down.weight", ".lora_A.weight")
        nk = nk.replace(".lora_up.weight", ".lora_B.weight")
        if nk.endswith(".lora_A.weight"):
            base = k.rsplit(".lora_", 1)[0]
            alpha = sd.get(base + ".alpha")
            if alpha is not None:
                v = v * (float(alpha) / v.shape[0])   # alpha/rank
        nuovo[nk] = v
    pipe.load_lora_weights(nuovo, adapter_name=adapter_name)

def LTX25(new_lora, width, height, fps, num_frames, prompt_ita,
          path_primo, path_ultimo):
    global pipe_video, old_lora_video

    import os
    os.environ["HF_HUB_OFFLINE"] = "1"   # PRIMA di qualsiasi import
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    REPO = r"C:\Users\User\.cache\huggingface\hub\models--Lightricks--LTX-2.5-Diffusers\snapshots\a97165959b05f5eb52a9f40f19c55957e334abe9"
    new_lora = '' if new_lora == 'nolora' else new_lora

    # Ricarica la pipe solo se è la prima volta o la LoRA è cambiata
    if pipe_video is None or new_lora != old_lora_video:
        if pipe_video is not None:
            del pipe_video
            pipe_video = None
            gc.collect()
            torch.cuda.empty_cache()

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            REPO,
            subfolder="transformer",
            quantization_config=bnb,
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe_video = LTX2ConditionPipeline.from_pretrained(
            REPO,
            transformer=transformer,
            dtype=torch.bfloat16,
            local_files_only=True,
        )

        if new_lora:
            path_loravideo = os.path.join('lora_ltx', f"{new_lora}.safetensors")
            if os.path.exists(path_loravideo):
                carica_lora_convertita(pipe_video, path_loravideo, 'loravideo')
                pipe_video.set_adapters(['loravideo'], adapter_weights=[0.8])
            else:
                print(f"LoRA non trovata: {path_loravideo}")

        quantize(pipe_video.text_encoder, weights=qfloat8)
        freeze(pipe_video.text_encoder)
        pipe_video.enable_model_cpu_offload()

        def libera_gpu(module, args, kwargs):
            for nome in ("connectors", "vae"):
                comp = getattr(pipe_video, nome)
                if next(comp.parameters()).device.type == "cuda":
                    comp.to("cpu")
            torch.cuda.empty_cache()

        pipe_video.transformer.register_forward_pre_hook(libera_gpu, with_kwargs=True)
        pipe_video.vae.enable_tiling()

        old_lora_video = new_lora      # memorizzato sempre, anche con nolora

    # ===== da qui in poi FUORI dall'if: gira a ogni generazione =====

    # --- parametri dai widget ---
    width, height = map(int, risoluzione.get().lower().split('x'))
    fps = int(fps_V.get())
    num_frames = int(frames_V.get())

    # --- prompt (traduzione IT -> EN) ---
    prompt_ita = text.get("1.0", "end-1c")
    print(f"PROMPT ITA: {prompt_ita}")
    try:
        prompt = traduci_prompt(prompt_ita)
    except Exception as e:
        print(f"Errore nella traduzione automatica: {e}")
        prompt = prompt_ita
    print(f"PROMPT EN: {prompt}")

    # --- immagini di condizione: carica, adatta (resize + crop centrale) ---
    first_image = carica_condizione(path_primo, ultimo_frame=True)
    last_image = carica_condizione(path_ultimo, ultimo_frame=False)

    if first_image is None and last_image is None:
        print("Seleziona almeno un fotogramma di riferimento")
        return

    if first_image is not None:
        first_image = fit_crop(first_image, width, height)
    if last_image is not None:
        last_image = fit_crop(last_image, width, height)

    conditions = []
    if first_image is not None:
        conditions.append(LTX2VideoCondition(frames=first_image, index=0, strength=1.0))
    if last_image is not None:
        conditions.append(LTX2VideoCondition(frames=last_image, index=-1, strength=1.0))

    print(f"Condizioni: {os.path.basename(path_primo or '-')} - "
          f"{os.path.basename(path_ultimo or '-')}")

    # --- generazione ---
    generator = torch.Generator("cpu").manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # risoluzioni predefinite LTX 2.5 (multipli di 32):
    # 256x160, 512x320, 704x384, 768x512, 960x544, 1024x576, 1280x704

    video, audio = pipe_video(
        conditions=conditions,
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=float(fps),
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        audio_guidance_scale=1.0,
        stg_scale=0.0,
        audio_stg_scale=0.0,
        modality_scale=1.0,
        audio_modality_scale=1.0,
        generator=generator,
        output_type="np",
        return_dict=False,
    )
    print(f"\nTempo: {time.time() - t0:.0f} s | "
          f"picco VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")

    # --- salvataggio ---
    os.makedirs('clips', exist_ok=True)
    path_out = percorso_libero(os.path.join('clips', 'ltx_0.mp4'))
    encode_video(
        video[0],
        fps=fps,
        audio=None,
        output_path=path_out,
    )
    print(f"Video salvato: {path_out}")
    return path_out


import threading as T
video_in_corso = False

import traceback

def worker_video(args):
    global video_in_corso
    try:
        LTX25(*args)
    except Exception as e:
        print(f"Errore generazione video: {e}")
        traceback.print_exc()
    finally:
        video_in_corso = False
        btn_Generavideo.config(text='Genera Video', state='normal')

def f_genera_video():
    global video_in_corso
    if video_in_corso:
        return
    video_in_corso = True

    # lettura dei widget nel thread principale
    width, height = map(int, risoluzione.get().lower().split('x'))
    args = (
        lora_video.get().strip(),
        width, height,
        int(fps_V.get()),
        int(frames_V.get()),
        text.get("1.0", "end-1c"),
        path_primo_fotogramma,
        path_ultimo_fotogramma,
    )
    btn_Generavideo.config(text='Generazione...', state='disabled')
    T.Thread(target=worker_video, args=(args,), daemon=True).start()

btn_Generavideo = tk.Button(frame_button_video, text='Genera Video', bg='#485afc',
                            command=f_genera_video)
btn_Generavideo.grid(row=0, column=4, sticky='w', padx=2, pady=2)

os.makedirs('lora_ltx', exist_ok=True)

def load_loraV(event=None):
    lora_files = [os.path.splitext(f)[0] for f in os.listdir('lora_ltx') if f.lower().endswith(('.safetensors', '.pt', '.ckpt'))]
    modelsV = ['nolora'] + sorted(lora_files)
    lora_video['values'] = modelsV
    if not lora_video.get() or lora_video.get() not in modelsV:
        lora_video.set(modelsV[0])

lab_lora=tk.Label(frame_button_video,text='Lora LTX 2.5').grid(row=0, column=5, sticky='w', padx=2, pady=2)
lora_video = ttk.Combobox(frame_button_video, values=['nolora'],width=8)
lora_video.grid(row=1, column=5, sticky='w', padx=2, pady=2)
load_loraV()
lora_video.bind('<Button-1>', load_loraV)
lora_video.current(0)

# risoluzioni video LTX 2.5
lab_risoluzionevideo = tk.Label(frame_button_video, text='Risoluzione Video')
lab_risoluzionevideo.grid(row=0, column=6, sticky='w', padx=2, pady=2)

risoluzione = ttk.Combobox(
    frame_button_video,
    values=[
        '256x160',    # 16:10 (il 16:9 esatto sarebbe 256x144, ma 144 non è multiplo di 32)
        '512x320',
        '704x384',    # ~16:9 (al posto di 720x*)
        '768x512',
        '960x544',
        '1024x576',
        '1280x704'
    ],
    state='readonly',
    width=12
)
risoluzione.current(3)  # 768x512 predefinito
risoluzione.grid(row=1, column=6, sticky='w', padx=2, pady=2)

lab_fps_V = tk.Label(frame_button_video, text='FPS')
lab_fps_V.grid(row=0, column=7, sticky='w', padx=2, pady=2)
fps_V = ttk.Combobox(frame_button_video, values=['8', '16', '24'], width=5, state='readonly')
fps_V.grid(row=1, column=7, sticky='w', padx=2, pady=2)
fps_V.set('24')

def f_load_frames(event=None):
    # frame validi 8n+1, da 49 a 121
    frames_V['values'] = [str(n) for n in range(49, 122, 8)]
    if not frames_V.get():
        frames_V.set('121')

lab_frames_V = tk.Label(frame_button_video, text='Frames per Clip')
lab_frames_V.grid(row=0, column=8)
frames_V = ttk.Combobox(frame_button_video, values=[], width=5, state='readonly')
frames_V.grid(row=1, column=8)

f_load_frames()
# Canvas iniziale (dopo che tutti i widget esistono)
aggiungi_canvas()
controlla_coda()   # avvia il polling della coda risultati

# tools
import threading as T
thr_motion=None
def run_motion():
    global thr_motion
    def ltx_m():
        os.system("python ltx25video.py")
    #controlla se il theding per avvio di ltx25video.py è gia avviato e gia in corso o è stato chiuso
    if thr_motion is None or not thr_motion.is_alive():
        thr_motion=T.Thread(target=ltx_m,daemon=True)
        thr_motion.start()
        print("avvio LTX 2.5 Motion Control")
    else:
        print("LTX Motion Control è gia aperto")
motion_control = tk.Button(frame_button_video, text='Motion Control', bg='#a449ff', command=run_motion)
motion_control.grid(row=1, column=9,padx=5)

import webbrowser

def run_A2E():
    webbrowser.open("https://video.a2e.ai/image-to-video")

A2E = tk.Button(frame_button_video, text='A2E', bg='#a4004d', fg='white', width=8,
                activebackground='#c2005c', activeforeground='white',
                font=('Segoe UI', 9, 'bold'), command=run_A2E)
A2E.grid(row=1, column=10,padx=5)

def run_Grok():
    webbrowser.open("https://grok.com/")

Grok = tk.Button(frame_button_video, text='Grok', bg='#000000', fg='white',width=8,
                 activebackground='#1a1a1a', activeforeground='white',
                 font=('Segoe UI', 9, 'bold'), command=run_Grok)
Grok.grid(row=1, column=11,padx=5)

from tkinter import messagebox
from moviepy import concatenate_videoclips, VideoFileClip

def f_rendering():
    global paths_image_canvas
    msg = messagebox.askquestion("Rendering", f"Render delle clips:\n{paths_image_canvas}\nProseguire?")
    if msg == 'yes':
        path_outfull = './videofull.mp4'
        k = 1
        while os.path.exists(path_outfull):
            path_outfull = f"./videofull_{k}.mp4"
            k += 1
        print("Rendering")
        clips = []
        for f in paths_image_canvas:
            if f.lower().endswith(('.mp4', '.mkv')):
                try:
                    video = VideoFileClip(f)
                    if clips and video.size != clips[0].size:
                        video = video.resized(new_size=clips[0].size)
                    clips.append(video)
                except Exception as e:
                    print(f"Errore nel caricare il video {f}: {e}")
        if clips:
            try:
                videofull = concatenate_videoclips(clips, method="compose")
                # Scrivi il file video. Codec e argomenti coerenti con MoviePy e ffmpeg
                videofull.write_videofile(
                    path_outfull,
                    codec='libx264',
                    audio=False,
                    fps=clips[0].fps,
                    threads=4
                )
                videofull.close()
            except Exception as e:
                print(f"Errore durante il rendering del video finale: {e}")
            finally:
                for video in clips:
                    video.close()
        else:
            print("Nessun video valido selezionato per il rendering.")
    else:
        print("Rendering annullato")

Rendering=tk.Button(frame_button_video, text='Rendering', bg='green', fg='white',width=8,
                 activebackground='#1a1a1a', activeforeground='white',
                 font=('Segoe UI', 9, 'bold'), command=f_rendering)
Rendering.grid(row=1, column=12,padx=5)

window.mainloop()