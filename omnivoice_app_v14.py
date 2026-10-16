# ===========================================================================
#  OmniVoice DEFINITIVO - app (motor oficial k2-fsa/OmniVoice)
#  v6.0: + le as vozes de uma pasta local (clonada do GitHub) em vez do
#        Google Drive. Acaba com a tela de autorizacao do Drive a cada
#        sessao nova do Colab. Nada mais foi alterado em relacao ao v5.9.
#  v5.9: + endereco FIXO via ngrok (dominio reservado da conta). Se o ngrok
#        falhar, cai no comportamento do v5.8 sem quebrar nada.
#  v5.8: + fallback pro proxy nativo do Colab quando o link publico do Gradio
#        falha (share=True sem tunnel) e cai no 127.0.0.1 (link morto).
#  v5.7: + "Baixar tudo (ZIP)" + lembra ultima voz/qualidade/expressividade.
#  v5.6: barra de progresso + ETA ao vivo + estimativa auto-calibrada + 'gerado em Xs'.
#  v5.5: vozes prontas com vocabulario VALIDO + 'Abrir em nova aba' + avisos amigaveis.
#  v5.2/5.3/5.4: site tela cheia, partes ~10 min, apagar voz, testar voz prontas,
#                estimativa de audio, slider Expressividade.
# ===========================================================================
import os, re, json, time, math, shutil, threading, zipfile
import numpy as np
import soundfile as sf
import torch
import gradio as gr
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

VERSAO       = "6.0"
MODEL_ID     = "k2-fsa/OmniVoice"
TEM_GPU      = torch.cuda.is_available()
DEVICE       = "cuda:0" if TEM_GPU else "cpu"
DTYPE        = torch.float16 if TEM_GPU else torch.float32
SR           = 24000
VOZES_DIR    = "/content/omnivoice_dados/vozes_salvas"
SAIDAS_DIR   = "/content/narracoes"
PREFS_FILE   = "/content/_prefs.json"
LIMITE_CHUNK = 300
PART_SECONDS = 600     # corta a narração em arquivos de ~10 minutos
FRASE_TESTE  = "Olá! Essa é uma amostra da minha voz, gerada pela OmniVoice."
os.makedirs(VOZES_DIR, exist_ok=True)
os.makedirs(SAIDAS_DIR, exist_ok=True)

if not TEM_GPU:
    print("\n⚠️  ATENÇÃO: nenhuma GPU detectada! Vá em "
          "'Ambiente de execução → Alterar tipo de ambiente → GPU T4' e rode tudo de novo.\n")

print(f"OmniVoice v{VERSAO} | motor oficial k2-fsa | dispositivo: {DEVICE}")

_model       = None
_lock        = threading.Lock()
ABORTAR      = threading.Event()
_clone_cache = {}   # (ref_audio, ref_text) -> VoiceClonePrompt (evita reprocessar)
_veloc       = {"seg_por_bloco": None}   # calibra a ETA com gerações reais


def _ler_prefs():
    try:
        with open(PREFS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _salvar_prefs(**kw):
    d = _ler_prefs()
    d.update(kw)
    try:
        with open(PREFS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
    except Exception:
        pass


PREFS = _ler_prefs()


def carregar_modelo():
    global _model
    if _model is None:
        print("Carregando o modelo OmniVoice... (1ª vez baixa ~3 GB, leva 1-2 min)")
        _model = OmniVoice.from_pretrained(
            MODEL_ID, device_map=DEVICE, dtype=DTYPE, load_asr=True,
        )
        print("Modelo OmniVoice pronto!")
    return _model


# --- Vozes prontas (voice design). O modelo só aceita "portuguese accent"
#     (= Portugal). Para sotaque BRASILEIRO use a aba Clonar. ---
MODELOS_PREDEFINIDOS = {
    "🧔 Narrador (grave)":     "male, middle-aged, low pitch, portuguese accent",
    "🎙️ Locutor (jovem)":      "male, young adult, portuguese accent",
    "👴 Velho sábio":          "male, elderly, low pitch, portuguese accent",
    "🤫 Sussurro masculino":   "male, whisper, portuguese accent",
    "👩 Apresentadora":        "female, young adult, portuguese accent",
    "📖 Narradora":            "female, middle-aged, portuguese accent",
    "🤫 Sussurro feminino":    "female, whisper, portuguese accent",
}
# Emoção = ajuste de TOM (único controle aceito pelo modelo no voice design).
SUFIXOS_EMOCAO = {
    "Neutro":    "",
    "Animado":   ", high pitch",
    "Sério":     ", low pitch",
    "Triste":    ", very low pitch",
    "Empolgado": ", very high pitch",
}
# Expressividade: ↑ valor = ↓ guidance_scale (voz mais solta/variada). 3 = padrão (2.0).
EXPRESS_MAP = {1: 3.0, 2: 2.5, 3: 2.0, 4: 1.6, 5: 1.3}


def _fmt_tempo(seg):
    seg = int(round(max(0, seg)))
    if seg < 60:
        return f"{seg}s"
    m, s = divmod(seg, 60)
    return f"{m} min {s}s"


def limpar_texto(texto):
    """Normaliza o texto para melhor pronúncia em pt-BR."""
    if not texto:
        return texto
    texto = re.sub(r"\n[ \t]*\n([ \t]*\n)+", "\n\n", texto)
    texto = "\n".join(l.rstrip() for l in texto.split("\n")).strip()

    def _caps(m):
        trecho = m.group(0)
        palavras = trecho.split()
        if len(palavras) == 1 and len(trecho) <= 4:
            return trecho            # sigla curta: mantém (PIB, FII...)
        return trecho.title()
    texto = re.sub(
        r"\b[A-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]{2,}(?:\s+[A-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]{2,})+\b", _caps, texto)
    return texto


def dividir_em_blocos(texto, limite=LIMITE_CHUNK):
    texto = (texto or "").strip()
    if not texto:
        return []
    if len(texto) <= limite:
        return [texto]
    partes = re.split(r"(?<=[.!?;:])\s+", texto)
    blocos, atual = [], ""
    for parte in partes:
        parte = parte.strip()
        if not parte:
            continue
        if len(atual) + len(parte) + 1 <= limite:
            atual = (atual + " " + parte).strip()
        else:
            if atual:
                blocos.append(atual)
            atual = ""
            if len(parte) > limite:
                sub = ""
                for palavra in parte.split():
                    if len(sub) + len(palavra) + 1 <= limite:
                        sub = (sub + " " + palavra).strip()
                    else:
                        if sub:
                            blocos.append(sub)
                        sub = palavra
                atual = sub
            else:
                atual = parte
    if atual:
        blocos.append(atual)
    return blocos


def _genconf(num_step, guidance=2.0):
    return OmniVoiceGenerationConfig(num_step=int(num_step), guidance_scale=float(guidance))


def _get_clone_prompt(ref_audio, ref_text):
    """Cria (e guarda em cache) o prompt de clonagem, processado UMA vez."""
    chave = (ref_audio, (ref_text or "").strip())
    if chave not in _clone_cache:
        modelo = carregar_modelo()
        rt = ref_text.strip() if ref_text and ref_text.strip() else None
        _clone_cache[chave] = modelo.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=rt)
    return _clone_cache[chave]


def sintetizar(texto, ref_audio=None, ref_text="", instruct=None, num_step=24,
               guidance=2.0, progresso=None):
    """Gera o áudio e devolve uma LISTA de partes (cada uma de até ~PART_SECONDS)."""
    with _lock:
        modelo = carregar_modelo()
        blocos = dividir_em_blocos(texto)
        if not blocos:
            raise gr.Error("Digite algum texto para gerar o áudio.")
        total = len(blocos)
        gc = _genconf(num_step, guidance)
        clone_prompt = _get_clone_prompt(ref_audio, ref_text) if ref_audio else None
        silencio = np.zeros(int(0.2 * SR), dtype=np.float32)
        segmentos = []   # cada bloco (com silêncio entre eles)
        t0 = time.time()
        for i, bloco in enumerate(blocos):
            if ABORTAR.is_set():
                return None, 0
            if progresso is not None:
                if i == 0:
                    progresso(0.0, desc=f"Gerando bloco 1/{total}...")
                else:
                    rest = (time.time() - t0) / i * (total - i)
                    progresso(i / total, desc=f"Bloco {i+1}/{total} · ~{_fmt_tempo(rest)} restantes")
            kwargs = {"text": bloco, "generation_config": gc}
            if clone_prompt is not None:
                kwargs["voice_clone_prompt"] = clone_prompt
            elif instruct:
                kwargs["instruct"] = instruct
            try:
                saida = modelo.generate(**kwargs)
            except Exception as e:
                raise gr.Error(f"Erro na geração: {e}")
            audio = saida[0] if isinstance(saida, (list, tuple)) else saida
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32).squeeze()
            segmentos.append(audio)
            if i < len(blocos) - 1:
                segmentos.append(silencio)
        if ABORTAR.is_set():
            return None, 0
        dt = time.time() - t0
        if total > 0 and dt > 0:
            _veloc["seg_por_bloco"] = dt / total   # calibra a estimativa
        if progresso is not None:
            progresso(0.99, desc="Juntando e salvando o áudio...")
        # empacota em partes de até ~PART_SECONDS (corta só em fim de bloco)
        partes, atual, dur = [], [], 0.0
        for seg in segmentos:
            atual.append(seg)
            dur += len(seg) / SR
            if dur >= PART_SECONDS:
                partes.append(np.concatenate(atual))
                atual, dur = [], 0.0
        if atual:
            sobra = sum(len(x) for x in atual) / SR
            if partes and sobra < 0.2 * PART_SECONDS:     # tira "rabinho" curto
                partes[-1] = np.concatenate([partes[-1]] + atual)
            else:
                partes.append(np.concatenate(atual))
        return partes, total


def salvar_wav(onda, prefixo="voz", no_drive=True):
    nome = f"{prefixo}_{int(time.time()*1000)}.wav"
    caminho = os.path.join("/tmp", nome)
    sf.write(caminho, onda, SR)
    if no_drive:
        try:
            shutil.copy(caminho, os.path.join(SAIDAS_DIR, nome))
        except Exception:
            pass  # nunca deixa a falha no Drive quebrar a geração
    return caminho


def _salvar_partes(partes, prefixo):
    """Salva partes no Drive; devolve (lista_arquivos, caminho_player, caminho_zip)."""
    total = len(partes)
    carimbo = int(time.time())
    arquivos = []
    for k, onda in enumerate(partes, 1):
        nome = f"{prefixo}_{carimbo}_parte{k}de{total}.wav"
        cam = os.path.join("/tmp", nome)
        sf.write(cam, onda, SR)
        try:
            shutil.copy(cam, os.path.join(SAIDAS_DIR, nome))
        except Exception:
            pass
        arquivos.append(cam)
    # player: ouve tudo junto (não vai pro Drive)
    full = partes[0] if total == 1 else np.concatenate(partes)
    player = salvar_wav(full, "preview_full", no_drive=False)
    # ZIP com todas as partes (só quando há mais de uma)
    zip_path = None
    if total > 1:
        zip_path = os.path.join("/tmp", f"{prefixo}_{carimbo}_narracao.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
            for a in arquivos:
                z.write(a, os.path.basename(a))
    return arquivos, player, zip_path


# ----------------------------- Vozes salvas --------------------------------
def listar_vozes_salvas():
    if not os.path.isdir(VOZES_DIR):
        return []
    vozes = []
    for arq in sorted(os.listdir(VOZES_DIR)):
        if arq.endswith(".json"):
            try:
                with open(os.path.join(VOZES_DIR, arq), encoding="utf-8") as f:
                    vozes.append("🎤 " + json.load(f)["nome"])
            except Exception:
                pass
    return vozes


def opcoes_vozes():
    return listar_vozes_salvas()


def caminho_ref_voz_salva(opcao):
    if not opcao or not opcao.startswith("🎤 "):
        return None
    nome = opcao[2:].strip().replace(" ", "_")
    c = os.path.join(VOZES_DIR, nome + ".wav")
    if os.path.exists(c):
        return c
    jp = os.path.join(VOZES_DIR, nome + ".json")
    if os.path.exists(jp):
        with open(jp, encoding="utf-8") as f:
            d = json.load(f)
        alt = os.path.join(VOZES_DIR, os.path.basename(d.get("arquivo", "")))
        if os.path.exists(alt):
            return alt
    return None


def _salvar_audio_voz(audio_file, nome_voz):
    nl = nome_voz.strip().replace(" ", "_")
    dest = os.path.join(VOZES_DIR, nl + ".wav")
    data, sr = sf.read(audio_file)
    sf.write(dest, data, sr)
    with open(os.path.join(VOZES_DIR, nl + ".json"), "w", encoding="utf-8") as f:
        json.dump({"nome": nome_voz.strip(), "arquivo": dest}, f, ensure_ascii=False)
    return opcoes_vozes()


def _zip_update(zip_path):
    if zip_path:
        return gr.update(value=zip_path, visible=True)
    return gr.update(visible=False)


# ------------------------------- Handlers ----------------------------------
def estimativa(texto):
    texto = texto or ""
    n = len(texto)
    if n == 0:
        return "📊 digite o texto para ver a estimativa"
    nb = len(dividir_em_blocos(texto))
    seg = n / 15.0
    partes = max(1, math.ceil(seg / PART_SECONDS))
    extra = ""
    if _veloc["seg_por_bloco"]:
        extra = f" · ⏱ ~{_fmt_tempo(nb * _veloc['seg_por_bloco'])} pra gerar"
    return f"📊 ~{_fmt_tempo(seg)} de áudio · {nb} bloco(s) · {partes} arquivo(s){extra}"


def aviso_truncamento(texto):
    return (limpar_texto(texto) if texto else gr.update()), ""


def gerar_clone(ref_audio, ref_text, texto, num_step, express, progress=gr.Progress()):
    if not ref_audio:
        gr.Warning("Envie um áudio de referência (3 a 10s) primeiro.")
        return gr.update(), "⚠️ Envie um áudio de referência (3 a 10s) primeiro.", gr.update(), gr.update(), gr.update(), gr.update()
    if not texto or not texto.strip():
        gr.Warning("Escreva o texto que você quer narrar.")
        return gr.update(), "⚠️ Escreva o texto que você quer narrar.", gr.update(), gr.update(), gr.update(), gr.update()
    ABORTAR.clear()
    t = time.time()
    partes, n = sintetizar(limpar_texto(texto), ref_audio=ref_audio, ref_text=ref_text,
                           num_step=num_step, guidance=EXPRESS_MAP.get(int(express), 2.0),
                           progresso=progress)
    if partes is None:
        return (None, "⛔ Geração cancelada.", gr.update(visible=False),
                gr.update(visible=False), gr.update(value=None, visible=False), gr.update(visible=False))
    arquivos, player, zip_path = _salvar_partes(partes, "clone")
    _salvar_prefs(num_step=int(num_step), express=int(express))
    dur = round(sum(len(p) for p in partes) / SR, 1)
    return (player, f"✅ {n} bloco(s) | {dur}s de áudio | gerado em {_fmt_tempo(time.time()-t)} "
            f"| {len(arquivos)} arquivo(s) de ~10min pra baixar",
            gr.update(visible=True), gr.update(visible=True),
            gr.update(value=arquivos, visible=True), _zip_update(zip_path))


def abortar_handler():
    ABORTAR.set()
    return "⛔ Cancelando após o bloco atual..."


def salvar_voz_handler(ref_audio, nome):
    if not ref_audio:
        gr.Warning("Não há áudio de referência para salvar.")
        return "⚠️ Não há áudio de referência para salvar.", gr.update()
    if not nome or not nome.strip():
        gr.Warning("Dê um nome para a voz antes de salvar.")
        return "⚠️ Dê um nome para a voz antes de salvar.", gr.update()
    todas = _salvar_audio_voz(ref_audio, nome)
    msg = f"💾 Voz '{nome.strip()}' salva! Veja na aba 🎤 Vozes Salvas."
    return msg, gr.update(choices=todas, value=("🎤 " + nome.strip()))


def importar_voz(audio_file, nome_voz):
    if not audio_file:
        gr.Warning("Selecione um arquivo de áudio (WAV ou MP3).")
        return "⚠️ Selecione um arquivo de áudio (WAV ou MP3).", gr.update()
    if not nome_voz or not nome_voz.strip():
        gr.Warning("Digite um nome para identificar esta voz.")
        return "⚠️ Digite um nome para identificar esta voz.", gr.update()
    todas = _salvar_audio_voz(audio_file, nome_voz)
    return f"✅ Voz '{nome_voz.strip()}' importada!", \
        gr.update(choices=todas, value=("🎤 " + nome_voz.strip()))


def apagar_voz(opcao):
    if not opcao or not opcao.startswith("🎤 "):
        gr.Warning("Selecione uma voz salva para apagar.")
        return gr.update(), "⚠️ Selecione uma voz salva (na lista) para apagar."
    nome = opcao[2:].strip().replace(" ", "_")
    apagou = False
    for ext in (".wav", ".json"):
        p = os.path.join(VOZES_DIR, nome + ext)
        if os.path.exists(p):
            try:
                os.remove(p)
                apagou = True
            except Exception:
                pass
    todas = opcoes_vozes()
    msg = f"🗑️ Voz '{opcao[2:].strip()}' apagada." if apagou else "Nada para apagar."
    return gr.update(choices=todas, value=todas[0] if todas else None), msg


def gerar_com_voz_salva(opcao, texto, num_step, express, progress=gr.Progress()):
    if not opcao:
        gr.Warning("Selecione uma voz salva na lista.")
        return gr.update(), "⚠️ Selecione uma voz salva na lista.", gr.update(), gr.update()
    if not texto or not texto.strip():
        gr.Warning("Escreva o texto que você quer narrar.")
        return gr.update(), "⚠️ Escreva o texto que você quer narrar.", gr.update(), gr.update()
    ref = caminho_ref_voz_salva(opcao)
    if not ref:
        gr.Warning("Arquivo da voz não encontrado.")
        return gr.update(), f"⚠️ Arquivo da voz não encontrado: {opcao}", gr.update(), gr.update()
    ABORTAR.clear()
    t = time.time()
    partes, n = sintetizar(limpar_texto(texto), ref_audio=ref, num_step=num_step,
                           guidance=EXPRESS_MAP.get(int(express), 2.0), progresso=progress)
    if partes is None:
        return None, "⛔ Geração cancelada.", gr.update(value=None, visible=False), gr.update(visible=False)
    arquivos, player, zip_path = _salvar_partes(partes, "salva")
    _salvar_prefs(num_step=int(num_step), express=int(express), voz=opcao)
    dur = round(sum(len(p) for p in partes) / SR, 1)
    return (player, f"✅ {opcao} | {n} bloco(s) | {dur}s | gerado em {_fmt_tempo(time.time()-t)} "
            f"| {len(arquivos)} arquivo(s)",
            gr.update(value=arquivos, visible=True), _zip_update(zip_path))


def testar_voz_salva(opcao, num_step, express, progress=gr.Progress()):
    if not opcao:
        gr.Warning("Selecione uma voz salva na lista.")
        return gr.update(), "⚠️ Selecione uma voz salva na lista."
    ref = caminho_ref_voz_salva(opcao)
    if not ref:
        gr.Warning("Arquivo da voz não encontrado.")
        return gr.update(), f"⚠️ Arquivo da voz não encontrado: {opcao}"
    ABORTAR.clear()
    partes, _ = sintetizar(FRASE_TESTE, ref_audio=ref, num_step=num_step,
                           guidance=EXPRESS_MAP.get(int(express), 2.0), progresso=progress)
    if not partes:
        return None, "⛔ Cancelado."
    return salvar_wav(partes[0], "teste", no_drive=False), f"✅ Prévia de {opcao}"


def gerar_pronta(opcao, emocao, texto, num_step, express, progress=gr.Progress()):
    if not opcao:
        gr.Warning("Escolha uma voz pronta.")
        return gr.update(), "⚠️ Escolha uma voz pronta.", gr.update(), gr.update()
    if not texto or not texto.strip():
        gr.Warning("Escreva o texto que você quer narrar.")
        return gr.update(), "⚠️ Escreva o texto que você quer narrar.", gr.update(), gr.update()
    instruct = MODELOS_PREDEFINIDOS.get(opcao, "") + SUFIXOS_EMOCAO.get(emocao, "")
    ABORTAR.clear()
    t = time.time()
    partes, n = sintetizar(limpar_texto(texto), instruct=instruct, num_step=num_step,
                           guidance=EXPRESS_MAP.get(int(express), 2.0), progresso=progress)
    if partes is None:
        return None, "⛔ Geração cancelada.", gr.update(value=None, visible=False), gr.update(visible=False)
    arquivos, player, zip_path = _salvar_partes(partes, "pronta")
    _salvar_prefs(num_step=int(num_step), express=int(express))
    dur = round(sum(len(p) for p in partes) / SR, 1)
    return (player, f"✅ {opcao} | {emocao} | {n} bloco(s) | {dur}s | gerado em {_fmt_tempo(time.time()-t)} "
            f"| {len(arquivos)} arquivo(s)",
            gr.update(value=arquivos, visible=True), _zip_update(zip_path))


def testar_pronta(opcao, emocao, num_step, express, progress=gr.Progress()):
    if not opcao:
        gr.Warning("Escolha uma voz pronta.")
        return gr.update(), "⚠️ Escolha uma voz pronta."
    instruct = MODELOS_PREDEFINIDOS.get(opcao, "") + SUFIXOS_EMOCAO.get(emocao, "")
    ABORTAR.clear()
    partes, _ = sintetizar(FRASE_TESTE, instruct=instruct, num_step=num_step,
                           guidance=EXPRESS_MAP.get(int(express), 2.0), progresso=progress)
    if not partes:
        return None, "⛔ Cancelado."
    return salvar_wav(partes[0], "teste_pronta", no_drive=False), f"✅ Prévia de {opcao} ({emocao})"


def atualizar_lista():
    todas = opcoes_vozes()
    return gr.update(choices=todas, value=todas[0] if todas else None)


def desligar_vm(armado):
    if not armado:
        return True, ("⚠️ Já baixou sua narração? Ao desligar, os áudios temporários do Colab "
                      "somem. Baixe a sua narração antes de desligar. "
                      "Se já baixou, clique DE NOVO em '⏹ Desligar' para confirmar.")
    def _k():
        time.sleep(1)
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception:
            pass
    threading.Thread(target=_k, daemon=True).start()
    return False, "Encerrando a sessão e liberando a GPU. Pode fechar a janela."


# --------------------------------- UI --------------------------------------
_NSTEP = int(PREFS.get("num_step", 24))
_EXPR  = int(PREFS.get("express", 3))
SLIDER = dict(minimum=8, maximum=48, step=4, value=_NSTEP,
              label="Qualidade  (8 = rápido  |  48 = máxima)  •  recomendado 24-32")
EXPRESS = dict(minimum=1, maximum=5, step=1, value=_EXPR,
               label="Expressividade  (3 = padrão  •  ↑ mais solto/variado  •  ↓ mais firme)")
FILE_KW = dict(label="⬇️ Baixar narração (arquivos de ~10 min)",
               file_count="multiple", interactive=False, visible=False)
_VOZES_INI = opcoes_vozes()
_VOZ_DEF = PREFS.get("voz") if PREFS.get("voz") in _VOZES_INI else (_VOZES_INI[0] if _VOZES_INI else None)

with gr.Blocks(title=f"OmniVoice v{VERSAO}", theme=gr.themes.Soft(
        primary_hue="purple", neutral_hue="slate")) as app:
    gr.Markdown(f"# 🎙️ OmniVoice v{VERSAO} — Gerador de Voz")
    gr.Markdown(f"**Motor oficial k2-fsa/OmniVoice | {DEVICE}** • "
                "vozes carregadas do GitHub • narrações ficam no Colab até você baixar")

    with gr.Tabs():
        # ---------------- ABA 1: CLONAR ----------------
        with gr.Tab("🗣️ Clonar a sua Voz"):
            gr.Markdown("Envie um áudio curto (3 a 10s) e o programa fala com aquela voz. "
                        "**Esse é o caminho que dá o sotaque brasileiro perfeito.**")
            clone_ref = gr.Audio(label="Áudio de referência (3 a 10 segundos)",
                                 sources=["upload"], type="filepath")
            clone_ref_text = gr.Textbox(
                label="Transcrição do áudio de referência (opcional — acelera e melhora)",
                placeholder="Deixe em branco que o programa transcreve sozinho.", lines=2)
            clone_texto = gr.Textbox(label="Texto para narrar",
                                     placeholder="Escreva aqui o que a voz deve falar...", lines=6)
            clone_estimativa = gr.Markdown("📊 digite o texto para ver a estimativa")
            num_step_c = gr.Slider(**SLIDER)
            express_c  = gr.Slider(**EXPRESS)
            with gr.Row():
                clone_botao   = gr.Button("▶ Gerar áudio", variant="primary")
                clone_abortar = gr.Button("⛔ Abortar", variant="stop")
            clone_status = gr.Textbox(label="Status", value="", interactive=False)
            clone_saida  = gr.Audio(label="Resultado (ouvir tudo)", type="filepath")
            clone_arquivos = gr.File(**FILE_KW)
            clone_zip = gr.DownloadButton("📦 Baixar tudo (ZIP)", variant="primary", visible=False)
            with gr.Row():
                clone_nome   = gr.Textbox(label="Nome da voz", placeholder="ex: Minha Voz Dark",
                                          visible=False, scale=3)
                clone_salvar = gr.Button("💾 Salvar esta voz", variant="secondary",
                                         visible=False, scale=1)
            clone_salvar_msg = gr.Markdown("")

        # ---------------- ABA 2: VOZES SALVAS ----------------
        with gr.Tab("🎤 Vozes Salvas"):
            gr.Markdown("Suas vozes, carregadas do GitHub (as mesmas em toda sessão).")
            with gr.Row():
                vozes_radio     = gr.Radio(choices=_VOZES_INI, value=_VOZ_DEF, label="Voz salva", scale=3)
                vozes_atualizar = gr.Button("🔄 Atualizar lista", scale=1)
                vozes_apagar    = gr.Button("🗑️ Apagar voz", variant="stop", scale=1)
            vozes_texto = gr.Textbox(label="Texto para narrar",
                                     placeholder="Escreva aqui o que a voz deve falar...", lines=6)
            vozes_estimativa = gr.Markdown("📊 digite o texto para ver a estimativa")
            num_step_v = gr.Slider(**SLIDER)
            express_v  = gr.Slider(**EXPRESS)
            with gr.Row():
                vozes_botao   = gr.Button("▶ Gerar áudio", variant="primary")
                vozes_testar  = gr.Button("Testar voz")
                vozes_abortar = gr.Button("⛔ Abortar", variant="stop")
            vozes_status = gr.Textbox(label="Status", value="", interactive=False)
            with gr.Row():
                vozes_saida = gr.Audio(label="Resultado (ouvir tudo)", type="filepath")
                vozes_teste = gr.Audio(label="Prévia", type="filepath")
            vozes_arquivos = gr.File(**FILE_KW)
            vozes_zip = gr.DownloadButton("📦 Baixar tudo (ZIP)", variant="primary", visible=False)

        # ---------------- ABA 3: VOZES PRONTAS ----------------
        with gr.Tab("🎚️ Vozes Prontas (Portugal)"):
            gr.Markdown("Vozes geradas por descrição (sem áudio). ⚠️ **O modelo só fala português "
                        "de PORTUGAL aqui** — para sotaque **brasileiro**, use a aba **🗣️ Clonar**.")
            pronta_radio  = gr.Radio(choices=list(MODELOS_PREDEFINIDOS.keys()),
                                     value="🧔 Narrador (grave)", label="Voz")
            pronta_emocao = gr.Radio(choices=list(SUFIXOS_EMOCAO.keys()),
                                     value="Neutro", label="Tom (emoção)")
            pronta_texto  = gr.Textbox(label="Texto para narrar",
                                       placeholder="Escreva aqui o que a voz deve falar...", lines=6)
            pronta_estimativa = gr.Markdown("📊 digite o texto para ver a estimativa")
            num_step_p = gr.Slider(**SLIDER)
            express_p  = gr.Slider(**EXPRESS)
            with gr.Row():
                pronta_botao   = gr.Button("▶ Gerar áudio", variant="primary")
                pronta_testar  = gr.Button("Testar voz")
                pronta_abortar = gr.Button("⛔ Abortar", variant="stop")
            pronta_status = gr.Textbox(label="Status", value="", interactive=False)
            with gr.Row():
                pronta_saida  = gr.Audio(label="Resultado (ouvir tudo)", type="filepath")
                pronta_teste  = gr.Audio(label="Prévia", type="filepath")
            pronta_arquivos = gr.File(**FILE_KW)
            pronta_zip = gr.DownloadButton("📦 Baixar tudo (ZIP)", variant="primary", visible=False)

        # ---------------- ABA 4: IMPORTAR ----------------
        with gr.Tab("📂 Importar Voz do PC"):
            gr.Markdown("### Importe um arquivo de voz do seu computador")
            with gr.Row():
                with gr.Column():
                    imp_audio  = gr.Audio(label="Arquivo WAV ou MP3", type="filepath",
                                          sources=["upload"])
                    imp_nome   = gr.Textbox(label="Nome da voz", placeholder="ex: Voz ARI")
                    btn_import = gr.Button("📥 Importar Voz", variant="primary")
                with gr.Column():
                    status_imp = gr.Textbox(label="Status", interactive=False, lines=3)
                    gr.Markdown("Depois, vá na aba **🎤 Vozes Salvas** e clique **🔄 Atualizar lista**.")

    with gr.Row():
        btn_desligar    = gr.Button("⏹ Desligar e liberar GPU (use ao terminar)", variant="stop")
        status_desligar = gr.Textbox(label="", interactive=False, scale=3)
    desliga_state = gr.State(False)
    gr.Markdown(f"*OmniVoice v{VERSAO} • motor k2-fsa • blocos até {LIMITE_CHUNK} chars • "
                f"narração cortada a cada ~{PART_SECONDS//60} min*")

    # ----- ligações -----
    clone_texto.change(fn=aviso_truncamento, inputs=clone_texto,
                       outputs=[clone_texto, clone_salvar_msg], show_progress="hidden")
    clone_texto.change(fn=estimativa, inputs=clone_texto, outputs=clone_estimativa,
                       show_progress="hidden")
    clone_botao.click(fn=gerar_clone,
                      inputs=[clone_ref, clone_ref_text, clone_texto, num_step_c, express_c],
                      outputs=[clone_saida, clone_status, clone_nome, clone_salvar, clone_arquivos, clone_zip])
    clone_abortar.click(fn=abortar_handler, outputs=clone_status)
    clone_salvar.click(fn=salvar_voz_handler, inputs=[clone_ref, clone_nome],
                       outputs=[clone_salvar_msg, vozes_radio])

    vozes_botao.click(fn=gerar_com_voz_salva, inputs=[vozes_radio, vozes_texto, num_step_v, express_v],
                      outputs=[vozes_saida, vozes_status, vozes_arquivos, vozes_zip])
    vozes_testar.click(fn=testar_voz_salva, inputs=[vozes_radio, num_step_v, express_v],
                       outputs=[vozes_teste, vozes_status])
    vozes_abortar.click(fn=abortar_handler, outputs=vozes_status)
    vozes_atualizar.click(fn=atualizar_lista, outputs=vozes_radio)
    vozes_apagar.click(fn=apagar_voz, inputs=vozes_radio, outputs=[vozes_radio, vozes_status])
    vozes_texto.change(fn=estimativa, inputs=vozes_texto, outputs=vozes_estimativa,
                       show_progress="hidden")

    pronta_botao.click(fn=gerar_pronta,
                       inputs=[pronta_radio, pronta_emocao, pronta_texto, num_step_p, express_p],
                       outputs=[pronta_saida, pronta_status, pronta_arquivos, pronta_zip])
    pronta_abortar.click(fn=abortar_handler, outputs=pronta_status)
    pronta_testar.click(fn=testar_pronta, inputs=[pronta_radio, pronta_emocao, num_step_p, express_p],
                        outputs=[pronta_teste, pronta_status])
    pronta_texto.change(fn=estimativa, inputs=pronta_texto, outputs=pronta_estimativa,
                        show_progress="hidden")

    btn_import.click(fn=importar_voz, inputs=[imp_audio, imp_nome],
                     outputs=[status_imp, vozes_radio])
    btn_desligar.click(fn=desligar_vm, inputs=desliga_state,
                       outputs=[desliga_state, status_desligar])

# ---- v5.9: endereco fixo via ngrok (opcional, nunca derruba o app) ----
NGROK_DOMINIO = "unmutteringly-wrongful-briella.ngrok-free.dev"

def _abrir_ngrok(demo):
    """Abre um tunel ngrok no dominio fixo. Devolve a URL, ou None se falhar."""
    try:
        import subprocess, sys as _sys
        try:
            from pyngrok import ngrok as _ng
        except ImportError:
            subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "pyngrok"], check=True)
            from pyngrok import ngrok as _ng
        from google.colab import userdata as _ud
        _tok = _ud.get("NGROK_TOKEN")
        if not _tok:
            print("AVISO: NGROK_TOKEN vazio nos Secrets do Colab - seguindo sem endereco fixo.")
            return None
        _ng.set_auth_token(_tok)
        try:
            _ng.kill()
        except Exception:
            pass
        _porta = getattr(demo, "server_port", None) or 7860
        _url = _ng.connect(_porta, domain=NGROK_DOMINIO).public_url
        print("OK: endereco fixo (ngrok) -> " + str(_url))
        return _url
    except Exception as _e:
        print("AVISO: ngrok nao subiu (" + str(_e) + ") - seguindo com o link normal do Gradio.")
        return None

# ---- abre como SITE (tela cheia), nao espremido dentro do Colab ----
carregar_modelo()
_demo = app.queue(default_concurrency_limit=1)
_res = _demo.launch(share=True, inline=False, prevent_thread_lock=True)
_fixo = _abrir_ngrok(_demo)   # v5.9: endereco fixo (None se o ngrok falhar)
_share = None
try:
    _share = _res[2] if isinstance(_res, (list, tuple)) and len(_res) > 2 else getattr(_demo, "share_url", None)
except Exception:
    _share = getattr(_demo, "share_url", None)
_share = _share or getattr(_demo, "share_url", None)
if not _share:
    # fallback: link publico do Gradio (share=True) falhou/nao gerou tunnel.
    # tenta o proxy nativo do proprio Colab (nao depende do tunnel do Gradio).
    try:
        from google.colab.output import eval_js as _colab_eval_js
        _port = None
        try:
            _srv = getattr(_demo, "server", None)
            _port = getattr(_srv, "port", None) or getattr(_demo, "server_port", None)
        except Exception:
            _port = getattr(_demo, "server_port", None)
        _port = _port or 7860
        _share = _colab_eval_js(f"google.colab.kernel.proxyPort({_port})")
    except Exception:
        _share = None
_share = _fixo or _share or getattr(_demo, "local_url", "")
try:
    from IPython.display import display, HTML
    display(HTML(
        '<div style="padding:16px;border:2px solid #7c3aed;border-radius:12px;'
        'background:#faf5ff;font-family:sans-serif;margin:10px 0;max-width:560px;">'
        '<b style="font-size:17px;">✅ OmniVoice pronto!</b><br><br>'
        '<a href="' + str(_share) + '" target="_blank" rel="noopener" '
        'style="display:inline-block;background:#7c3aed;color:#fff;padding:14px 24px;'
        'border-radius:10px;text-decoration:none;font-size:16px;font-weight:bold;">'
        '🎙️ Abrir OmniVoice em nova aba</a><br><br>'
        '<span style="color:#666;">Abre numa aba nova. Link: ' + str(_share) + '</span></div>'))
except Exception:
    pass
print("\n\n" + "=" * 66)
print("  ✅ APP PRONTO!  Clique no botão roxo acima (ou no link) para abrir.")
print("     👉  " + str(_share))
print("=" * 66 + "\n")
while True:        # mantém o app no ar; use o botão Desligar ao terminar
    time.sleep(3600)
