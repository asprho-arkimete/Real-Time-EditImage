import hashlib
import re
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from moviepy import VideoFileClip,ImageSequenceClip
from PIL import Image, ImageTk
import os

from transformers import MarianMTModel, MarianTokenizer

 

# Crea una finestra TkinterDnD
window = TkinterDnD.Tk()
window.title("LTX 2.5 Video")
window.geometry("1024x800")
window.resizable(False, False)
window.config(background='gray')


def mostra_immagine(canvas, img, size=200):
    """Ridimensiona un'immagine PIL mantenendo le proporzioni e la mostra nel canvas."""
    img = img.copy()
    img.thumbnail((size, size), Image.Resampling.BICUBIC)
    photo = ImageTk.PhotoImage(img)

    canvas.delete("all")  # rimuove testo e immagini precedenti
    canvas.create_image(size // 2, size // 2, image=photo, anchor='center')
    canvas.image_ref = photo  # evita che l'immagine venga eliminata dal garbage collector


from moviepy import VideoFileClip, ImageSequenceClip
from PIL import Image, ImageTk
import os

FPS = 24
ESTENSIONI_IMG = ('.jpg', '.jpeg', '.png')

out_path = None        # video da usare: quello caricato oppure quello temporaneo
file_path = None       # file originale trascinato (video o primo fotogramma)
tipo_sorgente = None   # "video" oppure "frames"


def crea_video_temp(dir_file, fps=FPS):
    """Crea un video temporaneo dalle immagini della cartella. Ritorna il percorso del video."""
    # Solo immagini, in ordine alfabetico
    files = sorted(
        os.path.join(dir_file, f)
        for f in os.listdir(dir_file)
        if f.lower().endswith(ESTENSIONI_IMG)
    )
    if not files:
        raise ValueError("Nessuna immagine nella cartella")
    else:
        for f in files:
            print(f"Ordine fotogrammi video Temp: {os.path.basename(f)} | ")

    # ImageSequenceClip richiede frame con le stesse dimensioni
    with Image.open(files[0]) as img:
        size = img.size
    validi = []
    for f in files:
        with Image.open(f) as img:
            if img.size == size:
                validi.append(f)
            else:
                print(f"Scartata (dimensioni diverse): {os.path.basename(f)}")

    nome_cartella = os.path.basename(os.path.normpath(dir_file))
    video_temp = os.path.join(dir_file, f"{nome_cartella}_temp.mp4")  # dentro la cartella dei fotogrammi
    clip = ImageSequenceClip(validi, fps=fps)
    try:
        try:
            clip.write_videofile(video_temp, codec="h264_nvenc", logger=None)  # GPU NVIDIA
        except Exception as e:
            print(f"GPU non disponibile ({e}), uso la CPU")
            clip.write_videofile(video_temp, codec="libx264", logger=None)
    finally:
        clip.close()
    return video_temp





def f_inputvideo(event):
    global out_path, file_path, tipo_sorgente, scorri_fotogrammi

    dropped = window.tk.splitlist(event.data)[0]

    if dropped.lower().endswith(('.mp4', '.mov', '.mkv')):
        try:
            with VideoFileClip(dropped) as clip:
                frame = clip.get_frame(0)
            first_image = Image.fromarray(frame)
            nuovo_out = dropped
            nuovo_tipo = "video"
            print("Primo frame video estratto.")
        except Exception as e:
            print(f"Errore nell'apertura del video: {e}")
            return
    elif dropped.lower().endswith(ESTENSIONI_IMG):
        try:
            first_image = Image.open(dropped)
            dir_file = os.path.dirname(dropped)
            nuovo_out = crea_video_temp(dir_file)
            nuovo_tipo = "frames"
            print(f"Video temporaneo creato in: {nuovo_out}")
        except Exception as e:
            print(f"Errore nell'elaborazione dell'immagine: {e}")
            return
    else:
        print("Formato non supportato")
        return

    file_path = dropped
    out_path = nuovo_out
    tipo_sorgente = nuovo_tipo

    # Aggiorno lo slider per entrambi i casi, leggendo il video da usare
    with VideoFileClip(out_path) as clip:
        n_frame = int(clip.duration * clip.fps)
    scorri_fotogrammi.config(to=max(n_frame, 0))
    scorri_fotogrammi.set(0)

    mostra_immagine(Input_video, first_image, 200)


frame_canvas = tk.Frame(window,bg='gray')
frame_canvas.grid(row=0, column=0, sticky='nw')

Input_video = tk.Canvas(frame_canvas, bg='red', width=200, height=200)
Input_video.create_text(100, 100, text="Trascina:", font=("Arial", 12), fill='white', anchor='center')
Input_video.create_text(100, 120, text="Video / Primo Frame", font=("Arial", 12), fill='white', anchor='center')
Input_video.grid(row=0, column=0, sticky='nw')

# Registra il canvas come destinazione del drop e collega l'evento
Input_video.drop_target_register(DND_FILES)
Input_video.dnd_bind('<<Drop>>', f_inputvideo)

from PIL import Image, ImageOps

path_ref = None  # file di riferimento trascinato
cropp = None     # immagine 768x512 ritagliata, da usare per l'elaborazione


def f_riferimento(event):
    global path_ref, cropp

    dropped = window.tk.splitlist(event.data)[0]

    if not dropped.lower().endswith(ESTENSIONI_IMG):
        print("Formato non supportato")
        return

    try:
        with Image.open(dropped) as img:
            # Ridimensiona a 768 di larghezza e ritaglia SOLO dal basso fino a 512 di altezza
            nuova_cropp = ImageOps.fit(
                img, (768, 512), Image.Resampling.BICUBIC, centering=(0.5, 0)
            )
    except Exception as e:
        print(f"Errore nell'apertura dell'immagine: {e}")
        return

    # Tutto è riuscito: aggiorno le globali
    path_ref = dropped
    cropp = nuova_cropp

    # Anteprima 128x128 nel canvas
    mostra_immagine(Input_primoframe, cropp, 128)


Input_primoframe = tk.Canvas(frame_canvas, bg='light blue', width=128, height=128)
Input_primoframe.create_text(64, 64, text="Trascina Immagine", font=("Arial", 12), fill='black', anchor='center')
Input_primoframe.grid(row=0, column=1, sticky='nw')

# Registra il canvas come destinazione del drop e collega l'evento
Input_primoframe.drop_target_register(DND_FILES)
Input_primoframe.dnd_bind('<<Drop>>', f_riferimento)

# bottoni estrai fotogrammi o taglia clips
frame_bottoni = tk.Frame(window,bg='gray')
frame_bottoni.grid(row=1, column=0, sticky='nw')

# Stato del video corrente (riempito da f_carica_video_scorrimento)


lab_scorri = tk.Label(frame_bottoni, text='Scorri Fotogrammi')
lab_scorri.grid(row=0, column=0, sticky='nw')

def f_scorri_fotogrammi(valore=None):      # <-- ora accetta il valore passato da Scale
    global Input_video, out_path
    if out_path is None:                   # nessun video ancora caricato
        return
    try:
        with VideoFileClip(out_path) as video:        # out_path, non file_path
            indice = int(scorri_fotogrammi.get())
            frame = video.get_frame(indice / video.fps)   # indice -> secondi

        image = Image.fromarray(frame)                # numpy -> PIL (ha .size e .resize)
        w, h = image.size
        rw, rh = 200, 200
        if w >= h:
            rh = (rw * h) // w
        else:
            rw = (rh * w) // h
        image = image.resize((rw, rh), Image.Resampling.BICUBIC)

        photo = ImageTk.PhotoImage(image)             # ImageTk, non tk.PhotoImage
        Input_video.delete("all")
        Input_video.create_image(100, 100, image=photo, anchor='center')
        Input_video.image_ref = photo                 # evita che l'immagine sparisca
        Input_video.update_idletasks()

        lab_info_frame.config(text=f"Fotogramma {indice}")
    except Exception as e:
        print(f"Errore scorrimento fotogrammi: {e}")

scorri_fotogrammi = tk.Scale(
    frame_bottoni, from_=0, to=100, orient=tk.HORIZONTAL,
    resolution=1, command=f_scorri_fotogrammi, length=100)
scorri_fotogrammi.grid(row=1, column=0, sticky='nw',padx=10,pady=5)

lab_info_frame = tk.Label(frame_bottoni, text='Fotogramma 0 / 0')
lab_info_frame.grid(row=2, column=0, sticky='nw',padx=10,pady=5)


def f_estraif():
    global out_path, file_path, scorri_fotogrammi
    print(f"Path: {out_path}")
    if not out_path or not os.path.exists(out_path):
        print("Nessun video caricato")
        return
    try:
        indice = int(scorri_fotogrammi.get())
        with VideoFileClip(out_path) as clip:
            frame = clip.get_frame(indice / clip.fps)   # indice -> secondi

        os.makedirs("clips", exist_ok=True)              # crea la cartella se manca
        nome = os.path.splitext(os.path.basename(out_path))[0].replace("_temp", "")
        path_save = os.path.join("clips", f"{nome}_frame{indice:05d}.png")

        Image.fromarray(frame).save(path_save)           # numpy -> PIL -> file
        print(f"Fotogramma salvato in: {path_save}")
    except Exception as e:
        print(f"Errore nell'estrazione del fotogramma: {e}")

estrai_frame = tk.Button(frame_bottoni, text='Estrai fotogramma', bg='light blue',
                         command=f_estraif)
estrai_frame.grid(row=1, column=1, sticky='nw')
def f_taglia_clip():
    global out_path, scorri_fotogrammi
    print(f"path: {out_path}")
    if not out_path or not os.path.exists(out_path):
        print("Nessun video caricato")
        return
    indice = int(scorri_fotogrammi.get())
    video = None
    clip_cut = None
    try:
        video = VideoFileClip(out_path)
        t_inizio = indice / video.fps                      # indice -> secondi
        clip_cut = video.subclipped(t_inizio)              # dal fotogramma scelto alla fine

        os.makedirs("clips", exist_ok=True)
        nome = os.path.splitext(os.path.basename(out_path))[0].replace("_temp", "")
        path_save = os.path.join("clips", f"{nome}_cut_frame{indice}.mp4")

        try:
            clip_cut.write_videofile(path_save, codec="h264_nvenc",
                                     audio_bitrate="128k", logger=None)   # GPU NVIDIA
        except Exception as e:
            print(f"GPU non disponibile ({e}), uso la CPU")
            clip_cut.write_videofile(path_save, codec="libx264",
                                     audio_bitrate="128k", logger=None)
        print(f"Clip salvata in: {path_save}")
    except Exception as e:
        print(f"Errore nel taglio della clip: {e}")
    finally:
        if clip_cut is not None:
            clip_cut.close()
        if video is not None:
            video.close()

taglia_clip = tk.Button(frame_bottoni, text='Taglia clip', bg='orange',
                        command=f_taglia_clip)
taglia_clip.grid(row=1, column=2, sticky='nw')



frame_text=tk.Frame(window,bg='light gray')
frame_text.grid(row=2,column=0)

lab_prompt = tk.Label(frame_text, text='Prompt Video')
lab_prompt.grid(row=0, column=0, sticky='nw',padx=10,pady=5)
text = tk.Text(frame_text, width=70, height=6)
text.grid(row=1, column=0, sticky='nw',padx=10,pady=5)

PROMPT_base = (
    """Use the reference video for motion, timing, camera movement, composition, 
and unchanged scene content, while consistently propagating the visual edit established 
in the first frame throughout the video.\n
Scena:""")

text.insert('1.0',PROMPT_base)

import os
import threading
import traceback
from itertools import islice
import gc
import torch
from PIL import Image, ImageOps
from moviepy import VideoFileClip
from diffusers import BitsAndBytesConfig, FlowMatchEulerDiscreteScheduler, LTX2InContextPipeline, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video
from optimum.quanto import freeze, qfloat8, quantize

MAX_FRAMES = 121                 # se serve ancora meno memoria, prova 97 (8n+1)
WIDTH, HEIGHT = 480, 320         # multipli di 32, rapporto 3:2 (alternativa: 512, 352)
FPS_OUT = 24
LORA_STRENGTH = 1.35
LORA2_STRENGTH = 0.8

REPO = r"C:\Users\User\.cache\huggingface\hub\models--Lightricks--LTX-2.5-Diffusers\snapshots\a97165959b05f5eb52a9f40f19c55957e334abe9"
PATH_NSFW = r"G:\realTime_create\model\transformer_nsfw"
DIR_LORA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_ltx")
PATH_LORA = os.path.join(DIR_LORA, "LTX25_Ripple_v11.safetensors")

pipe_video = None
modello_caricato = None  # 'originale' o 'NSFW': modello con cui è stata costruita pipe_video
lora2_attiva = None      # nome file della LoRA 2 attualmente caricata (None = nessuna)
ultima_firma = None      # firma degli input dell'ultima generazione riuscita
ultimo_output = None     # percorso dell'ultimo video generato


def fit_frame(img):
    """Stesso ritaglio per riferimento e frame del video: tiene la parte alta, taglia sotto."""
    return ImageOps.fit(img.convert("RGB"), (WIDTH, HEIGHT), Image.Resampling.BICUBIC, centering=(0.5, 0))


def impostazioni_modello(modello):
    """Parametri di generazione: originale = distillato, NSFW = modello completo (dev)."""
    if modello == 'originale':
        return dict(
            sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            stg_scale=0.0,
            modality_scale=1.0,
            audio_guidance_scale=1.0,
            audio_stg_scale=0.0,
            audio_modality_scale=1.0,
        )
    if modello == 'NSFW':
        return dict(
            num_inference_steps=30,
            guidance_scale=3.0,
            stg_scale=1.0,
            modality_scale=3.0,
            audio_guidance_scale=7.0,
            audio_stg_scale=1.0,
            audio_modality_scale=3.0,
        )
    raise ValueError(f"Modello sconosciuto: {modello}")


def applica_lora(pipe):
    """Allinea gli adapter alla selezione attuale (LoRA 2 e pesi). Va chiamata a ogni generazione."""
    global lora2_attiva

    nome = lora2.get().strip()
    if nome in ("", "nolora"):
        nome = None
    if nome is not None and not os.path.exists(os.path.join(DIR_LORA, nome)):
        raise FileNotFoundError(f"LoRA 2 non trovata: {os.path.join(DIR_LORA, nome)}")

    # LoRA 2 cambiata: rimuovi la vecchia e carica la nuova
    if nome != lora2_attiva:
        if lora2_attiva is not None:
            pipe.delete_adapters("lora2")
        if nome is not None:
            pipe.load_lora_weights(os.path.join(DIR_LORA, nome), adapter_name="lora2")
        lora2_attiva = nome

    # Una sola chiamata con tutti gli adapter attivi
    nomi = ["lora1"]
    pesi = [float(modifica1.get())]
    if lora2_attiva is not None:
        nomi.append("lora2")
        pesi.append(float(modifica2.get()))
    pipe.set_adapters(nomi, adapter_weights=pesi)


def scarica_pipeline():
    """Libera la pipeline in memoria (usata quando si cambia modello)."""
    global pipe_video, modello_caricato, lora2_attiva
    pipe_video = None
    modello_caricato = None
    lora2_attiva = None
    gc.collect()
    torch.cuda.empty_cache()


def carica_pipeline():
    """Carica il modello la prima volta o quando cambia; LoRA e pesi vengono sempre riallineati."""
    global pipe_video, modello_caricato

    modello = model_ltx.get()
    if pipe_video is not None and modello != modello_caricato:
        print(f"Cambio modello: {modello_caricato} -> {modello}")
        scarica_pipeline()

    if pipe_video is not None:
        applica_lora(pipe_video)
        return pipe_video
    if not os.path.exists(PATH_LORA):
        raise FileNotFoundError(f"LoRA non trovata: {PATH_LORA}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if modello == 'originale':
        print("Modello originale (distillato)")
        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            REPO, subfolder="transformer", quantization_config=bnb,
            dtype=torch.bfloat16, local_files_only=True,
        )
    elif modello == 'NSFW':
        print("Modello NSFW (completo)")
        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            PATH_NSFW, quantization_config=bnb,
            dtype=torch.bfloat16, local_files_only=True,
        )
    else:
        raise ValueError(f"Modello sconosciuto: {modello}")

    pipe = LTX2InContextPipeline.from_pretrained(      # pipeline IC-LoRA
        REPO, transformer=transformer, dtype=torch.bfloat16, local_files_only=True,
    )

    if modello == 'NSFW':
        # lo scheduler del repo è configurato per il distillato: per il modello completo serve lo shifting
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config, use_dynamic_shifting=True, shift_terminal=0.1
        )

    pipe.load_lora_weights(PATH_LORA, adapter_name="lora1")
    applica_lora(pipe)          # carica lora2 se selezionata e attiva gli adapter

    quantize(pipe.text_encoder, weights=qfloat8)
    freeze(pipe.text_encoder)
    pipe.enable_model_cpu_offload()

    def libera_gpu(module, args, kwargs):
        for nome in ("connectors", "vae"):
            comp = getattr(pipe, nome)
            if next(comp.parameters()).device.type == "cuda":
                comp.to("cpu")
        torch.cuda.empty_cache()

    pipe.transformer.register_forward_pre_hook(libera_gpu, with_kwargs=True)
    pipe.vae.enable_tiling()

    pipe_video = pipe
    modello_caricato = modello
    return pipe_video

# ---------------- Traduzione IT -> EN ----------------
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


def costruisci_prompt(contenuto):
    """Base (già in inglese) + scena scritta dall'utente (tradotta in inglese)."""
    contenuto = contenuto.strip()
    base_pulita = PROMPT_base.replace("Scena:", "").strip()
    base_tk = PROMPT_base.strip()

    if not contenuto:
        return base_pulita
    if contenuto.startswith(base_tk):
        scena = contenuto[len(base_tk):].strip()
        if not scena:
            return base_pulita
        return f"{base_pulita}\nScene: {traduci_prompt(scena)}"
    # l'utente ha cancellato/riscritto la base: traduco tutto il testo
    return traduci_prompt(contenuto)


# ---------------- Generazione ----------------
def f_ltx(testo_prompt):
    global ultima_firma, ultimo_output

    if out_path is None:
        print("Carica prima un video o una sequenza di frame")
        return
    if cropp is None:
        print("Carica prima l'immagine di riferimento")
        return

    print(f"Sorgente: {tipo_sorgente} | video: {out_path} | riferimento: {path_ref}")

    # Frame modificato (primo frame), stesso ritaglio dei frame del video
    edited = fit_frame(cropp)
    edited.save("temp_ref.png")  # per verifica

    # Firma degli input: video, immagine di riferimento, prompt
    firma = hashlib.md5(
        "|".join([
            str(out_path),
            str(os.path.getmtime(out_path)),                 # video modificato?
            hashlib.md5(edited.tobytes()).hexdigest(),       # riferimento cambiato?
            testo_prompt.strip(),                            # prompt cambiato?
        ]).encode()
    ).hexdigest()

    # Video sorgente a 24 fps, max MAX_FRAMES frame, tutti ritagliati allo stesso modo
    try:
        with VideoFileClip(out_path) as clip:
            frames = [fit_frame(Image.fromarray(f))
                      for f in islice(clip.iter_frames(fps=FPS_OUT), MAX_FRAMES)]
    except Exception as e:
        print(f"Errore nella lettura del video: {e}")
        return

    if len(frames) < 2:
        print("Video troppo corto")
        return

    # Come la demo: il frame modificato viene ANTEPOSTO, il frame 0 originale resta all'indice 1
    combined = [edited] + frames

    # Numero di frame del tipo 8n+1, max MAX_FRAMES
    num_frames = (min(len(combined), MAX_FRAMES) - 1) // 8 * 8 + 1
    if num_frames < 9:
        print("Servono almeno 9 fotogrammi")
        return
    combined = combined[:num_frames]
    print(f"Fotogrammi: {num_frames}")

    PROMPT_eng = costruisci_prompt(testo_prompt)
    print(f"Prompt: {PROMPT_eng}")

    pipe = carica_pipeline()     # carica la prima volta o se è cambiato il modello, poi riusa quella in memoria
    impostazioni = impostazioni_modello(modello_caricato)
    print(f"Modello: {modello_caricato} | impostazioni: {impostazioni.keys()}")

    video_out, audio = pipe(
        prompt=PROMPT_eng,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        reference_conditions=[LTX2ReferenceCondition(frames=combined, strength=1.0)],
        height=HEIGHT,
        width=WIDTH,
        num_frames=num_frames,
        frame_rate=float(FPS_OUT),
        generator=torch.Generator("cuda").manual_seed(42),
        output_type="np",
        return_dict=False,
        **impostazioni,
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))   # G:\realTime_create
    out_dir = os.path.join(base_dir, "clips")
    os.makedirs(out_dir, exist_ok=True)

    path_outvideo = os.path.join(out_dir, "ltx25.mp4")
    k = 1
    while os.path.exists(path_outvideo):
        path_outvideo = os.path.join(out_dir, f"ltx25-{k}.mp4")
        k += 1

    encode_video(
        video_out[0],
        fps=FPS_OUT,
        output_path=path_outvideo,
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    )

    # Salvo la firma solo se la generazione è andata a buon fine
    ultima_firma = firma
    ultimo_output = path_outvideo
    print(f"Video generato: {path_outvideo}")


def f_ltx_click():
    """Lancia la generazione in un thread, così l'interfaccia non si blocca."""
    ltx.config(state="disabled")
    testo_prompt = text.get('1.0', 'end-1c')     # lettura del widget nel thread della GUI

    def lavoro():
        try:
            f_ltx(testo_prompt)
        except Exception:
            traceback.print_exc()
        finally:
            window.after(0, lambda: ltx.config(state="normal"))

    threading.Thread(target=lavoro, daemon=True).start()

frame_canvas.config(background='gray')
frame_buttoni = tk.Frame(frame_canvas, background='gray')
frame_buttoni.grid(row=0, column=2, sticky='nw')
ltx = tk.Button(frame_buttoni, text='Genera', bg='light green', command=f_ltx_click)
ltx.grid(row=0, column=0, sticky='nw', padx=10, pady=5)

from tkinter import HORIZONTAL, ttk

frame_canvas.config(background='gray')
frame_buttoni = tk.Frame(frame_canvas, background='gray')
frame_buttoni.grid(row=0, column=2, sticky='nw')
ltx = tk.Button(frame_buttoni, text='Genera', bg='light green', command=f_ltx_click)
ltx.grid(row=1, column=0, sticky='nw', padx=10, pady=5)

model_ltx=ttk.Combobox(frame_buttoni,values=['originale','NSFW'],width=11)
model_ltx.grid(row=1, column=1, sticky='nw', padx=5, pady=5)
model_ltx.set('NSFW')
# Lora 1
lab_lora1 = tk.Label(frame_buttoni, text='Lora 1', background='gray')
lab_lora1.grid(row=0, column=2, sticky='nw', padx=2, pady=5)
lora1 = ttk.Combobox(frame_buttoni, values=['Ripple_v11'], width=15)
lora1.grid(row=1, column=2, sticky='nw', padx=2, pady=5)
lora1.set('Ripple_v11')
modifica1 = tk.Scale(frame_buttoni, from_=0.10, to=2.0, resolution=0.10, orient=HORIZONTAL)
modifica1.set(1.35)
modifica1.grid(row=2, column=2, sticky='nw', padx=2, pady=5)


# Lora 2 dynamic loading
def load_lora(event=None):
    try:
        models = ['nolora'] + sorted(
            f for f in os.listdir(DIR_LORA)
            if f.endswith('.safetensors') and f != os.path.basename(PATH_LORA)
        )
        lora2['values'] = models
    except Exception:
        lora2['values'] = ['nolora']
        lora2.set('nolora')

lab_lora2 = tk.Label(frame_buttoni, text='Lora 2', background='gray')
lab_lora2.grid(row=0, column=3, sticky='nw', padx=2, pady=5)
lora2 = ttk.Combobox(frame_buttoni, values=['nolora'], width=15, state='readonly')
lora2.grid(row=1, column=3, sticky='nw', padx=2, pady=5)
lora2.set('nolora')
lora2.bind('<Button-1>', load_lora)
modifica2 = tk.Scale(frame_buttoni, from_=0.10, to=2.0, resolution=0.10, orient=HORIZONTAL)
modifica2.set(0.7)
modifica2.grid(row=2, column=3, sticky='nw', padx=2, pady=5)




window.mainloop()