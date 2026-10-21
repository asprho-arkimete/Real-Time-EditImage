# Real-Time-EditImage

Questo progetto utilizza **FLUX.2** con 2 step di prova nella generazione delle immagini: in questo modo restituisce una bozza in tempo reale a bassa risoluzione e, successivamente, crea un'immagine ad alta risoluzione.

Contiene inoltre **LTX-2.5 quantizzato** per animare le immagini e una **timeline di montaggio e rendering**.

## Requisiti

- Windows
- GPU NVIDIA con CUDA
- Python 3.10
- Spazio su disco: circa 230 GB (FLUX.2 ~40 GB + LTX-2.5 ~190 GB)

## Installazione

### 1. Clona il repository

```bash
git clone https://github.com/asprho-arkimete/Real-Time-EditImage.git
cd Real-Time-EditImage
```

### 2. Crea e attiva l'ambiente virtuale (Python 3.10)

```bash
python -m venv vtr
vtr\Scripts\activate
```

### 3. Installa PyTorch con CUDA

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

### 4. Installa le librerie

```bash
pip install -r requisiti.txt
```

### 5. Primo avvio

Avvia `realtime.py` per creare le cartelle necessarie, poi chiudi il programma:

```bash
python realtime.py
```

### 6. Scarica i modelli e i file aggiuntivi

| Cosa | Dove scaricarlo | Dove metterlo |
|---|---|---|
| Modello LTX-2.5 | [colinb83/ltx-2.5-nsfw-step7000](https://huggingface.co/colinb83/ltx-2.5-nsfw-step7000) | cartella `model` |
| LoRA video | vedi il file `LTX_nsfw_istruzioni_e_download.txt` | cartella `lora_ltx` (creata al primo avvio) |
| LoRA di FLUX.2 | [Asprho/megalora](https://huggingface.co/Asprho/megalora/tree/main) | cartella `lora` |
| File posa | sezione **Releases** del repository | decomprimilo (unzip) |

### 7. Converti il modello LTX-2.5 in formato diffusers

Converti il file scaricato in formato diffusers con lo script `convert.py`.

### 8. Avvia il programma

```bash
python realtime.py
```

I modelli FLUX.2 e LTX-2.5 vengono scaricati automaticamente al primo utilizzo.

## Note

- Se non riesci a scaricare il modello LTX-2.5, rimuovi `local_files_only=True` dalla funzione `carica_pipeline()` (riga 407 di `realtime.py`).

- ## Risoluzione problemi

**Errore al caricamento di LTX-2.5 / il modello non si scarica**

Il programma carica LTX-2.5 solo dai file già presenti sul disco
(`local_files_only=True`). Se il modello non è ancora stato scaricato,
il caricamento fallisce. In questo caso:

1. Apri `realtime.py` e cerca la funzione `carica_pipeline()` (riga 407).
2. Elimina `local_files_only=True` dalla chiamata `from_pretrained(...)`.
3. Riavvia il programma: il modello viene scaricato in automatico.
4. A download finito puoi rimettere `local_files_only=True`
   per evitare controlli di rete a ogni avvio.



 
 
 
 
 
 
