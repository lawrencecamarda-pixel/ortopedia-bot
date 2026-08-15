import os
import glob
import logging
import asyncio
import pickle
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

try:
    import fitz  # PyMuPDF per estrazione ultra-veloce e robusta
except ImportError:
    fitz = None

from pypdf import PdfReader

# Carica variabili d'ambiente (.env)
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "OPENAI").upper()  # OPENAI, DEEPSEEK, GEMINI, OPENROUTER
API_KEY = os.getenv("API_KEY", "")
BOOKS_DIR = os.getenv("BOOKS_DIR", "./books")
CACHE_FILE = "library_index.pkl"

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Server HTTP per Health Check di Render Web Service (Free Tier)
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    
    def log_message(self, format, *args):
        pass  # Silenzia i log dell'Health Check

def start_health_check_server():
    try:
        port = int(os.getenv("PORT", "8080"))
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        logger.info(f"🌐 Server Health Check di Render attivo sulla porta {port}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Errore avvio server Health Check: {e}")

# Index dei libri
library_index: List[Dict] = []

def extract_pdf_paragraphs(pdf_path: str) -> List[Dict]:
    """Estrae paragrafi significativi da un PDF usando PyMuPDF (o pypdf come fallback)."""
    full_chunks = []
    filename = os.path.basename(pdf_path)

    if fitz is not None:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text() or ""
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
                for p in paragraphs:
                    full_chunks.append({
                        "text": p,
                        "page": page_num + 1,
                        "source": filename
                    })
            return full_chunks
        except Exception as e:
            logger.warning(f"PyMuPDF fallito per {filename}, uso pypdf: {e}")

    try:
        reader = PdfReader(pdf_path, strict=False)
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
            for p in paragraphs:
                full_chunks.append({
                    "text": p,
                    "page": page_num,
                    "source": filename
                })
    except Exception as e:
        logger.error(f"Errore lettura PDF {filename}: {e}")

    return full_chunks

def index_pdf_books(folder_path: str, force_reindex: bool = False):
    """Indicizza i libri PDF estratti ed imposta la cache per il caricamento istantaneo."""
    global library_index
    import gzip
    
    for cfile in ["library_index.pkl.gz", "library_index.pkl"]:
        if not force_reindex and os.path.exists(cfile):
            try:
                open_fn = gzip.open if cfile.endswith(".gz") else open
                with open_fn(cfile, "rb") as f:
                    library_index = pickle.load(f)
                logger.info(f"⚡ Caricato indice libreria da {cfile} ({len(library_index)} file elaborati).")
                return
            except Exception as e:
                logger.warning(f"Impossibile leggere {cfile}: {e}")

    library_index = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Cartella {folder_path} creata.")
        return

    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    logger.info(f"📖 Trovati {len(pdf_files)} libri PDF in {folder_path}. Avvio elaborazione...")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        logger.info(f"Elaborazione in corso: {filename}...")
        chunks = extract_pdf_paragraphs(pdf_path)
        if chunks:
            library_index.append({
                "title": filename,
                "chunks": chunks
            })
            logger.info(f"✅ Indicizzato: {filename} ({len(chunks)} paragrafi salvati)")

    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(library_index, f)
        logger.info("💾 Indice salvato su disco in 'library_index.pkl'.")
    except Exception as e:
        logger.error(f"Errore salvataggio cache: {e}")

HIGH_SPECIFICITY_TERMS = {
    "femore": ["femur", "femoral", "femoral neck", "distal femur", "proximal femur", "shaft"],
    "femorale": ["femoral", "femur", "distal femur", "femoral condyle", "femoral neck"],
    "femorali": ["femoral", "femur", "condyles"],
    "collo": ["femoral neck", "neck fracture", "pauwels", "garden"],
    "trocantere": ["trochanter", "trochanteric", "intertrochanteric", "subtrochanteric"],
    "pertrocanterica": ["intertrochanteric", "pertrochanteric", "trochanteric"],
    "sottotrocanterica": ["subtrochanteric"],
    "condilo": ["femoral condyle", "condylar", "distal femur", "supracondylar", "intercondylar", "condyle"],
    "condili": ["femoral condyles", "condylar", "distal femur", "supracondylar", "intercondylar", "condyles"],
    "condilare": ["condylar", "distal femur", "supracondylar"],
    "condilari": ["condylar", "distal femur", "supracondylar"],
    "sovracondilica": ["supracondylar", "distal femur"],
    "supracondiloidea": ["supracondylar", "distal femur"],
    "anca": ["hip", "hip joint", "femoral head", "arthroplasty"],
    "testa": ["femoral head", "head fracture", "pipkin"],
    "cotile": ["acetabulum", "acetabular", "judet", "letournel"],
    "acetabolo": ["acetabulum", "acetabular", "judet", "letournel"],
    "bacino": ["pelvis", "pelvic", "pelvic ring", "tile", "young-burgess"],
    "tibia": ["tibia", "tibial", "tibial shaft", "distal tibia", "proximal tibia"],
    "tibiale": ["tibial", "tibia", "tibial plateau", "tibial spine", "tibial shaft"],
    "tibiali": ["tibial", "tibia", "plateau", "spines"],
    "piatto": ["tibial plateau", "schatzker", "plateau fracture"],
    "pilon": ["pilon", "plafond", "distal tibia"],
    "spine": ["tibial spine", "intercondylar eminence", "tibial eminence", "meyers", "mckeever", "spine"],
    "spina": ["tibial spine", "intercondylar eminence", "tibial eminence", "meyers", "mckeever"],
    "eminematica": ["intercondylar eminence", "tibial eminence"],
    "tuberosita": ["tuberosity", "tibial tubercle"],
    "perone": ["fibula", "fibular", "fibular head", "lateral malleolus"],
    "fibula": ["fibula", "fibular", "malleolus"],
    "ginocchio": ["knee", "patella", "patellar", "extensor mechanism", "acl", "pcl"],
    "rotula": ["patella", "patellar", "extensor mechanism"],
    "patella": ["patella", "patellar"],
    "lca": ["acl", "anterior cruciate"],
    "lcp": ["pcl", "posterior cruciate"],
    "menisco": ["meniscus", "meniscal", "root tear", "bucket handle"],
    "meniscale": ["meniscus", "meniscal", "root tear"],
    "menischi": ["meniscus", "meniscal"],
    "root": ["root tear", "meniscal root"],
    "lfpb": ["mpfl", "patellofemoral"],
    "caviglia": ["ankle", "malleolar", "lauge-hansen", "weber"],
    "malleolo": ["malleolus", "malleolar", "bimalleolar", "trimalleolar", "weber"],
    "malleolare": ["malleolus", "malleolar", "weber"],
    "malleoli": ["malleoli", "malleolar"],
    "calcagno": ["calcaneus", "calcaneal", "sanders"],
    "astragalo": ["talus", "talar", "hawkins"],
    "lisfranc": ["lisfranc", "tarsometatarsal"],
    "chopart": ["chopart", "midtarsal"],
    "metatarso": ["metatarsal", "forefoot"],
    "spalla": ["shoulder", "glenohumeral", "rotator cuff", "bankart", "hill-sachs"],
    "omero": ["humerus", "humeral", "proximal humerus", "humeral shaft", "distal humerus"],
    "omerale": ["humeral", "humerus", "proximal humerus"],
    "scapola": ["scapula", "scapular", "glenoid"],
    "glena": ["glenoid", "glenoid fossa"],
    "clavicola": ["clavicle", "clavicular", "acromioclavicular"],
    "cuffia": ["rotator cuff", "supraspinatus"],
    "gomito": ["elbow", "olecranon", "radial head", "coronoid"],
    "olecrano": ["olecranon"],
    "capitello": ["radial head", "mason"],
    "radio": ["radius", "radial", "distal radius", "radial head", "shaft"],
    "radiale": ["radial", "radius", "distal radius"],
    "ulna": ["ulna", "ulnar", "olecranon"],
    "polso": ["wrist", "distal radius", "scaphoid", "cooney", "fernandez"],
    "scafoide": ["scaphoid", "carpal scaphoid"],
    "carpo": ["carpus", "carpal"],
    "metacarpo": ["metacarpal", "hand"],
    "colonna": ["spine", "spinal", "vertebra", "vertebral"],
    "rachide": ["spine", "spinal", "vertebral"],
    "cervicale": ["cervical", "cervical spine"],
    "dorsale": ["thoracic", "thoracic spine"],
    "lombare": ["lumbar", "lumbar spine"],
    "sacro": ["sacrum", "sacral", "sacroiliac"],
    "frattura": ["fracture", "fractures", "broken"],
    "fratture": ["fracture", "fractures"],
    "esposta": ["open fracture", "gustilo", "anderson"],
    "esposte": ["open fractures", "gustilo"],
    "scomposta": ["displaced", "displacement"],
    "comminuta": ["comminuted", "comminution"],
    "articolare": ["articular", "intra-articular"],
    "diafisi": ["diaphysis", "shaft", "diaphyseal"],
    "diafisaria": ["diaphysis", "shaft", "diaphyseal"],
    "metafisi": ["metaphysis", "metaphyseal"],
    "epifisi": ["epiphysis", "epiphyseal", "salter-harris"],
    "pseudoartrosi": ["nonunion", "malunion", "pseudoarthrosis", "delayed union"],
    "ao": ["ao", "muller", "classification"],
    "schatzker": ["schatzker"],
    "garden": ["garden"],
    "pauwels": ["pauwels"],
    "neer": ["neer"],
    "rockwood": ["rockwood"],
    "meyers": ["meyers", "mckeever"],
    "mason": ["mason"],
    "weber": ["weber", "lauge-hansen"]
}

def search_relevant_chunks(query: str, top_k: int = 10) -> List[Dict]:
    """Algoritmo di ricerca clinica bilingue ad alta precisione con supporto a tutte le regioni ortopediche."""
    if not library_index:
        return []
    
    query_clean = query.lower()
    query_words = re.findall(r'\w+', query_clean)
    
    specific_keywords = set()
    for w in query_words:
        if w in HIGH_SPECIFICITY_TERMS:
            specific_keywords.update(HIGH_SPECIFICITY_TERMS[w])
        elif len(w) > 3:
            specific_keywords.add(w)

    scored_chunks = []

    for book in library_index:
        book_title = book["title"].lower()
        for chunk in book["chunks"]:
            chunk_text = chunk["text"].lower()
            score = 0
            
            for word in query_words:
                if len(word) > 3 and word in chunk_text:
                    score += 1
            
            for spec_term in specific_keywords:
                if spec_term in chunk_text:
                    score += 6
            
            if any(k in query_clean for k in ["spine", "ginocchio", "knee", "lca", "menisco", "condilo"]):
                if "insall" in book_title or "knee" in book_title or "brotzman" in book_title:
                    score += 3
            if any(k in query_clean for k in ["frattura", "fratture", "trauma", "femore", "tibia", "omero", "condilo"]):
                if "rockwood" in book_title or "ao" in book_title or "traumatology" in book_title:
                    score += 3

            if score > 0:
                scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_chunks[:top_k]]

async def query_llm(prompt: str, context: str) -> str:
    """Interrogazione asincrona al modello AI con retry automatico in caso di rate-limit."""
    system_prompt = (
        "Sei un assistente esperto in Ortopedia e Traumatologia riservato a medici specializzandi.\n"
        "Rispondi al quesito clinico, teorico o di trattamento basandoti sul materiale allegato estratto dai libri e protocolli di ortopedia.\n"
        "Mantieni un tono medico professionale, chiaro, rigoroso e strutturato (es. Eziologia, Classificazione, Indicazioni Chirurgiche/Conservative).\n"
        "Se il testo allegato non contiene informazioni sufficienti, segnalalo chiaramente."
    )
    
    user_content = f"### Estratti dai libri e protocolli di Ortopedia:\n{context}\n\n### Domanda dello specializzando:\n{prompt}"

    for attempt in range(3):
        try:
            if LLM_PROVIDER in ["OPENAI", "DEEPSEEK", "OPENROUTER"]:
                from openai import AsyncOpenAI
                
                base_url = None
                model_name = "gpt-4o-mini"
                
                if LLM_PROVIDER == "DEEPSEEK":
                    base_url = "https://api.deepseek.com"
                    model_name = "deepseek-chat"
                elif LLM_PROVIDER == "OPENROUTER":
                    base_url = "https://openrouter.ai/api/v1"
                    model_name = "google/gemini-2.0-flash-001"
                
                client = AsyncOpenAI(api_key=API_KEY, base_url=base_url)
                
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.2,
                    max_tokens=1200
                )
                return response.choices[0].message.content.strip()

            elif LLM_PROVIDER == "GEMINI":
                import google.generativeai as genai
                genai.configure(api_key=API_KEY)
                model = genai.GenerativeModel('gemini-flash-latest')
                full_prompt = f"{system_prompt}\n\n{user_content}"
                response = await asyncio.to_thread(model.generate_content, full_prompt)
                return response.text.strip()

        except Exception as e:
            err_str = str(e)
            logger.warning(f"Tentativo LLM {attempt + 1} fallito: {err_str}")
            if "429" in err_str or "quota" in err_str.lower():
                await asyncio.sleep(4)
            elif attempt == 2:
                return f"⚠️ Si è verificato un errore temporaneo nelle API: {err_str}"

    return "⚠️ Impossibile ottenere risposta dal servizio AI dopo diversi tentativi."

# Handlers Telegram
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🩺 *Bot Ortopedia & Traumatologia Policlinico*\n\n"
        "Benvenuto! Il bot è collegato alla tua libreria di testi e protocolli di ortopedia.\n\n"
        "Puoi fare domande su trattamenti, classificazioni (AO, Schatzker, Neer, Rockwood, Garden, Pauwels, Meyers & McKeever), indicazioni chirurgiche e riabilitative.\n\n"
        "📌 *Comandi disponibili:*\n"
        "/start - Mostra questo messaggio\n"
        "/libri - Elenco dei manuali e protocolli caricati\n"
        "/ricarica - Re-indicizza la cartella se hai aggiunto nuovi libri"
    )
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def libri_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not library_index:
        await update.message.reply_text("📚 Nessun libro caricato.")
        return
    
    text = "📚 *Testi e Protocolli disponibili in libreria:*\n\n"
    for b in library_index:
        text += f"• `{b['title']}` ({len(b['chunks'])} paragrafi indicizzati)\n"
    await update.message.reply_text(text, parse_mode='Markdown')

async def ricarica_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Avvio aggiornamento ed indicizzazione dei libri...")
    index_pdf_books(BOOKS_DIR, force_reindex=True)
    await update.message.reply_text(f"✅ Aggiornamento completato! {len(library_index)} testi pronti alla consultazione.")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_query = update.message.text

    await update.message.chat.send_action("typing")

    relevant_chunks = search_relevant_chunks(user_query, top_k=10)
    
    if not relevant_chunks:
        context_str = "Nessun estratto trovato nei testi. Rispondi con la tua conoscenza generale ortopedica specificando che non c'è riscontro diretto nei testi della libreria."
    else:
        context_str = ""
        for chunk in relevant_chunks:
            context_str += f"\n--- [Da: {chunk['source']} - Pagina {chunk['page']}] ---\n{chunk['text']}\n"

    reply_text = await query_llm(user_query, context_str)

    if relevant_chunks:
        sources_used = sorted(list(set(f"{c['source']} (pag. {c['page']})" for c in relevant_chunks[:5])))
        reply_text += "\n\n📖 *Riferimenti estratti dalla tua libreria:* \n" + "\n".join([f"• _{s}_" for s in sources_used])

    try:
        await update.message.reply_text(reply_text, parse_mode='Markdown')
    except Exception:
        await update.message.reply_text(reply_text, parse_mode=None)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Eccezione Telegram intercettata: {context.error}")

def main():
    # Avvia il server di Health Check per Render Web Service (Free Tier)
    threading.Thread(target=start_health_check_server, daemon=True).start()

    print("📚 Caricamento libreria ortopedica...")
    index_pdf_books(BOOKS_DIR, force_reindex=False)

    if not TELEGRAM_TOKEN:
        print("❌ ERRORE: TELEGRAM_TOKEN mancante nel file .env!")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("libri", libri_handler))
    app.add_handler(CommandHandler("ricarica", ricarica_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    app.add_error_handler(error_handler)

    print("🤖 Bot Telegram attivo ed in ascolto per gli specializzandi!")
    app.run_polling()

if __name__ == "__main__":
    main()
