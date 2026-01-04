import chainlit as cl
from chainlit.input_widget import Slider
import ollama
import PyPDF2
import io

@cl.on_chat_start
async def baslangic():
    # --- DEĞİŞİKLİK 1: SİSTEM MESAJI SADELEŞTİRİLDİ ---
    # Artık modele "şunu yap, bunu yapma" diye yalvarmamıza gerek yok.
    # Ham model (ham-deepseek) zaten doğası gereği <think> etiketiyle başlıyor.
    system_message = {
        "role": "system", 
        "content": "You are a helpful assistant. Always show your reasoning step-by-step."
    } 
    
    cl.user_session.set("message_history", [system_message])
    
    settings = await cl.ChatSettings(
        [
            Slider(id="Temperature", label="Temperature", initial=0.7, min=0, max=1, step=0.1)
        ]
    ).send()
    
    # --- DEĞİŞİKLİK 2: MODEL İSMİ GÜNCELLENDİ ---
    # Ollama'nın filtrelerini bypass eden kendi oluşturduğumuz modeli kullanıyoruz.
    cl.user_session.set("model", "ham-deepseek")
    cl.user_session.set("temperature", 0.7)
    
    # Mesajı güncelledik
    await cl.Message(content="🚀 **Raw DeepSeek (ham-deepseek) Hazır!**").send()

@cl.on_settings_update
async def ayarlar_degisti(settings):
    cl.user_session.set("temperature", settings["Temperature"])

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content
    model = cl.user_session.get("model")
    current_temp = cl.user_session.get("temperature")
    message_history = cl.user_session.get("message_history")
    
    dosya_icerigi = ""
    
    # --- GELİŞMİŞ DOSYA OKUMA MOTORU (V3.0) --- 
    # (Bu kısım harika çalıştığı için dokunmadım, aynen korundu)
    if message.elements:
        processing_msg = cl.Message(content="📂 The file is being analyzed...")
        await processing_msg.send()
        
        for element in message.elements:
            try:
                if element.content:
                    file_bytes = element.content
                elif element.path:
                    with open(element.path, "rb") as f:
                        file_bytes = f.read()
                else:
                    raise ValueError("The file content or path could not be found!")

                if "text" in element.mime:
                    try:
                        text = file_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        text = file_bytes.decode("latin-1")
                    dosya_icerigi += f"\n--- DOSYA: {element.name} ---\n{text}\n"
                
                elif "pdf" in element.mime:
                    pdf_file = io.BytesIO(file_bytes)
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted
                    
                    if not text.strip():
                        await cl.Message(content=f"⚠️ **Warning:** '{element.name}' was read but no text was found.").send()
                    else:
                        dosya_icerigi += f"\n--- DOSYA: {element.name} ---\n{text}\n"
                
                print(f"DEBUG: {element.name} read successfully.")

            except Exception as e:
                print(f"HATA: {e}")
                await cl.Message(content=f"❌ File cannot be read: {element.name}\nTechnical error: {str(e)}").send()

        processing_msg.content = f"✅ **File contents retrieved.** Response being generated..."
        await processing_msg.update()

    # --- PROMPT HAZIRLIĞI ---
    if dosya_icerigi:
        final_prompt = f"Analyze the document below and answer the question accordingly.:\n\nDOKÜMAN İÇERİĞİ:\n{dosya_icerigi}\n\nKULLANICI SORUSU: {user_input}"
    else:
        final_prompt = user_input

    message_history.append({"role": "user", "content": final_prompt})

    msg = cl.Message(content="")

    async with cl.Step(name="Thought Process", type="process") as step:
        step.input = user_input
        
        stream = ollama.chat(
            model=model,
            messages=message_history,
            options={'temperature': current_temp},
            stream=True
        )

        # --- DEĞİŞİKLİK 3: DAHA GÜVENLİ AKIŞ MANTIĞI ---
        # Önceki 'buffer' mantığını 'full_response' ile değiştirdim.
        # Bu yöntem token parçalanmalarına (örn: '<' ve 'think>' ayrı gelirse) karşı daha garantidir.
        
        full_response = "" 
        thought_ended = False 

        for chunk in stream:
            token = chunk['message']['content']
            
            # Gelen her şeyi havuzda biriktiriyoruz
            full_response += token 

            if not thought_ended:
                # Henüz düşünme bitmediyse kontrol et: Etiket kapandı mı?
                if "</think>" in full_response:
                    thought_ended = True
                    
                    # Metni tam ortadan ikiye bölüyoruz
                    parts = full_response.split("</think>")
                    thought_part = parts[0].replace("<think>", "").strip()
                    answer_part = parts[1] if len(parts) > 1 else ""
                    
                    # 1. Düşünce kutusunu güncelle ve kapat
                    step.output = thought_part
                    await step.update()
                    
                    # 2. Eğer cevap kısmı geldiyse hemen ana mesaja bas
                    if answer_part:
                        await msg.stream_token(answer_part)
                else:
                    # Düşünme devam ediyor...
                    # <think> yazısını ekranda göstermemek için filtreleyip kutuya yazıyoruz
                    display_token = token.replace("<think>", "")
                    await step.stream_token(display_token)
            else:
                # Düşünme bitti, artık gelen her şey cevaptır. Direkt bas.
                await msg.stream_token(token)
    
    # --- DEĞİŞİKLİK 4: GEÇMİŞİ KAYDETME ---
    # Önceki kodda final_answer değişkeni karışabiliyordu. 
    # Artık full_response değişkeni her şeyi tuttuğu için onu direkt kaydediyoruz.
    message_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("message_history", message_history)
    
    await msg.send()