import os
import io
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Se si modifica la portata (scopes), cancellare il file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def sync_drive_folder(folder_id: str, output_dir: str = "./books"):
    """Scarica tutti i PDF ed i documenti di testo presenti nella cartella Google Drive specificata."""
    os.makedirs(output_dir, exist_ok=True)
    creds = None
    
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif os.path.exists('credentials.json'):
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        else:
            print("⚠️ File 'credentials.json' non trovato. Scarica le credenziali OAuth 2.0 da Google Cloud Console.")
            return

    try:
        service = build('drive', 'v3', credentials=creds)
        # Cerca file nella cartella specifica
        query = f"'{folder_id}' in parents and mimeType = 'application/pdf' and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print("Nessun file PDF trovato nella cartella Google Drive.")
            return

        print(f"Trovati {len(items)} file PDF su Google Drive. Inizio il download...")
        for item in items:
            file_id = item['id']
            file_name = item['name']
            file_path = os.path.join(output_dir, file_name)

            if os.path.exists(file_path):
                print(f"⏭️ File già presente locale: {file_name}")
                continue

            print(f"📥 Download in corso: {file_name}...")
            request = service.files().get_media(fileId=file_id)
            fh = io.FileIO(file_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"   Avanzamento: {int(status.progress() * 100)}%")
            print(f"✅ Download completato: {file_name}")

    except Exception as e:
        print(f"❌ Errore durante il collegamento a Google Drive: {e}")

if __name__ == "__main__":
    folder_id = input("Inserisci l'ID della cartella Google Drive (quello finale nell'URL): ").strip()
    if folder_id:
        sync_drive_folder(folder_id)
    else:
        print("ID cartella non valido.")
