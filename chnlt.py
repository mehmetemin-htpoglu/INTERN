import chainlit as cl
from chainlit.input_widget import Slider
import ollama
import PyPDF2
import io

@cl.on_chat_start
async def baslangic():
    # System Prompt: Modeli her durumda <think> etiketlerini kullanmaya zorluyoruz
    system_message = {
        "role": "system", 
        "content": (
            "You are a helpful assistant. You must always behave as a reasoning model. "
            "CRITICAL RULE: Every single response must start with a thought process enclosed within <think> and </think> tags. "
            "If the user query is simple and requires no reasoning, you must still output <think> </think> (with a space inside) before providing your final answer. "
            "Never provide an answer without these tags."
        )
    } 
    # Message history'yi bu sistem mesajıyla başlatıyoruz
    cl.user_session.set("message_history", [system_message])
    
    # Ayarlar
    settings = await cl.ChatSettings(
        [
            Slider(id="Temperature", label="Temperature", initial=0.7, min=0, max=1, step=0.1)
        ]
    ).send()
    
    # Session degiskenleri
    cl.user_session.set("model", "deepseek-r1:14b")
    cl.user_session.set("temperature", 0.7)
    
    # Acilis mesaji
    await cl.Message(content="DeepSeek-R1:14B is ready!").send()

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
    if message.elements:
        processing_msg = cl.Message(content="📂 The file is being analyzed...")
        await processing_msg.send()
        
        for element in message.elements:
            try:
                # 1. ADIM: Dosya verisini al (RAM'de yoksa Diskten oku)
                if element.content:
                    file_bytes = element.content
                elif element.path:
                    # Chainlit dosyayı diske kaydettiyse oradan okuyoruz
                    with open(element.path, "rb") as f:
                        file_bytes = f.read()
                else:
                    raise ValueError("The file content or path could not be found!")

                # 2. ADIM: Dosya Türüne Göre İşle
                # Metin Dosyası (.txt, .py, .md vb.)
                if "text" in element.mime:
                    try:
                        text = file_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        text = file_bytes.decode("latin-1") # Türkçe karakter kurtarıcısı
                        
                    dosya_icerigi += f"\n--- DOSYA: {element.name} ---\n{text}\n"
                
                # PDF Dosyası
                elif "pdf" in element.mime:
                    pdf_file = io.BytesIO(file_bytes)
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted
                    
                    if not text.strip():
                        await cl.Message(content=f"⚠️ **Warning:** '{element.name}' was read but no text was found (It might be an image).").send()
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

    # Hafızaya ekle
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

        final_answer = "" 
        is_thinking = True 
        buffer = "" # Etiketi yakalamak için geçici hafıza

        for chunk in stream:
            token = chunk['message']['content']

            if is_thinking:
                buffer += token # Gelen parçaları birleştiriyoruz
                
                # Tamponda kapanış etiketi var mı diye kontrol et
                if "</think>" in buffer:
                    is_thinking = False
                    
                    # Tamponu parçala: düşünce kısmı ve cevap kısmı
                    parts = buffer.split("</think>")
                    thought_content = parts[0].replace("<think>", "").strip()
                    first_answer_part = parts[1] if len(parts) > 1 else ""
                    
                    # Düşünce adımını tamamla ve güncelle (Ekrana temiz basar)
                    step.output = thought_content
                    await step.update()
                    
                    # Eğer etiketin hemen peşinden cevap geldiyse onu ana mesaja bas
                    if first_answer_part:
                        final_answer += first_answer_part
                        await msg.stream_token(first_answer_part)
                else:
                    # Henüz etiket kapanmadıysa buffer'daki son eklenen kısmı canlı akıtmak yerine
                    # sadece düşünce sürecine eklemeye devam edebiliriz.
                    # Ancak canlı görünmesi için şimdilik token'ı step'e akıtabiliriz.
                    # Not: Canlı akışta <think> yazısı görünebilir, son update ile temizlenir.
                    clean_token = token.replace("<think>", "")
                    await step.stream_token(clean_token)
            else:
                # Artık düşünme bitti, doğrudan ana cevaba yaz
                final_answer += token
                await msg.stream_token(token)
    
    message_history.append({"role": "assistant", "content": final_answer})
    cl.user_session.set("message_history", message_history)
    
    await msg.send()