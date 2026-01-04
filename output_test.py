import chainlit as cl
import ollama

@cl.on_chat_start
async def start():
    # Test için gerekli ayarları yüklüyoruz
    cl.user_session.set("model", "deepseek-r1:14b")
    cl.user_session.set("temperature", 0.7)
    # Boş bir geçmişle başlayalım
    cl.user_session.set("message_history", [])
    
    await cl.Message(content="🕵️ **Dedektif Modu Aktif!**\nLütfen bir soru sor ve VS Code terminalini izle.").send()

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content
    model = cl.user_session.get("model")
    current_temp = cl.user_session.get("temperature")
    message_history = cl.user_session.get("message_history")
    
    # Mesajı geçmişe ekle
    message_history.append({"role": "user", "content": user_input})

    msg = cl.Message(content="")
    
    # Terminalde başlangıcı işaretleyelim
    print(f"\n{'='*20} STREAM BAŞLIYOR (HAM VERİ) {'='*20}") 

    stream = ollama.chat(
        model=model,
        messages=message_history,
        options={'temperature': current_temp},
        stream=True
    )

    full_response = ""

    for chunk in stream:
        token = chunk['message']['content']
        
        # 1. HAM VERİYİ TERMINALE BAS (Filtresiz)
        # flush=True ile anında yazmasını sağlıyoruz, bekleme yapmaz
        print(token, end="", flush=True) 
        
        full_response += token
        
        # 2. ARAYÜZE BAS (Chainlit ne yapıyor görelim)
        await msg.stream_token(token)
    
    print(f"\n{'='*20} STREAM BİTTİ {'='*20}\n")
    
    message_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("message_history", message_history)
    
    await msg.send()