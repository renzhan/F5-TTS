import base64
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# 添加src目录到Python路径
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# 延迟导入F5TTS
F5TTS = None

# 默认模型配置
DEFAULT_MODEL = "F5TTS_v1_Base"
DEFAULT_CKPT_FILE = "models/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
DEFAULT_VOCAB_FILE = "models/F5-TTS/F5TTS_v1_Base/vocab.txt"

# 全局变量存储模型实例
f5tts_instance = None
current_model = None
current_ckpt_file = None
model_loaded = False

def lazy_import_f5tts():
    """延迟导入F5TTS模块"""
    global F5TTS
    if F5TTS is None:
        try:
            print("📦 正在导入F5TTS模块...")
            from f5_tts.api import F5TTS
            print("✅ F5TTS模块导入成功")
        except Exception as e:
            print(f"❌ F5TTS模块导入失败: {e}")
            raise e
    return F5TTS

def get_f5tts_instance(model: str, ckpt_file: str = "", vocab_file: str = ""):
    """获取或创建F5TTS实例"""
    global f5tts_instance, current_model, current_ckpt_file, model_loaded
    
    if (f5tts_instance is None or 
        current_model != model or 
        current_ckpt_file != ckpt_file):
        
        try:
            # 延迟导入F5TTS
            F5TTS_class = lazy_import_f5tts()
            
            print(f"🔄 正在加载模型: {model}")
            print(f"📁 模型文件: {ckpt_file}")
            print(f"📝 词汇表: {vocab_file}")
            
            f5tts_instance = F5TTS_class(
                model=model,
                ckpt_file=ckpt_file,
                vocab_file=vocab_file
            )
            current_model = model
            current_ckpt_file = ckpt_file
            model_loaded = True
            
            print(f"✅ 模型加载成功: {model}")
            print(f"📍 模型路径: {ckpt_file}")
        except Exception as e:
            model_loaded = False
            print(f"❌ 模型加载失败: {e}")
            raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")
    
    return f5tts_instance

def preload_model():
    """预加载默认模型（可选）"""
    try:
        print("🚀 开始预加载F5-TTS模型...")
        get_f5tts_instance(DEFAULT_MODEL, DEFAULT_CKPT_FILE, DEFAULT_VOCAB_FILE)
        print("🎉 模型预加载完成！")
    except Exception as e:
        print(f"⚠️  模型预加载失败: {e}")
        print("💡 将在首次请求时尝试加载模型")

# 创建FastAPI应用
app = FastAPI(
    title="F5-TTS API",
    description="F5-TTS 文本转语音 API 服务",
    version="1.0.0"
)

# 可选的启动事件，用于预加载模型
# 注释掉以避免启动时卡住
# @app.on_event("startup")
# async def startup_event():
#     """启动时的事件处理"""
#     print("🌟 F5-TTS API 服务启动中...")
#     preload_model()

@app.get("/")
async def root():
    """根路径，返回API信息"""
    global model_loaded, current_model, current_ckpt_file
    
    return {
        "message": "F5-TTS API 服务正在运行",
        "version": "1.0.0",
        "model": {
            "name": current_model or DEFAULT_MODEL,
            "path": current_ckpt_file or DEFAULT_CKPT_FILE,
            "loaded": model_loaded,
            "status": "已加载" if model_loaded else "未加载"
        },
        "endpoints": {
            "POST /tts-simple": "简化版文本转语音（使用默认参数）",
            "POST /preload": "预加载模型",
            "GET /models": "获取支持的模型列表",
            "GET /health": "健康检查"
        }
    }

@app.post("/preload")
async def preload_endpoint():
    """手动预加载模型的端点"""
    try:
        preload_model()
        return {
            "message": "模型预加载成功",
            "model_loaded": model_loaded,
            "model": current_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型预加载失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查"""
    global model_loaded
    return {
        "status": "healthy" if model_loaded else "degraded",
        "message": "F5-TTS API 服务正常运行" if model_loaded else "服务运行中，但模型未加载",
        "model_loaded": model_loaded
    }

@app.get("/models")
async def get_models():
    """获取支持的模型列表"""
    global model_loaded, current_model
    
    models = [
        {
            "name": "F5TTS_v1_Base",
            "description": "F5-TTS v1 Base模型",
            "ckpt_path": DEFAULT_CKPT_FILE,
            "vocab_path": DEFAULT_VOCAB_FILE,
            "status": "已加载" if (model_loaded and current_model == "F5TTS_v1_Base") else "可用",
            "is_default": True
        }
    ]
    return {"models": models}

@app.post("/tts-simple")
async def simple_tts(
    ref_audio: UploadFile = File(..., description="参考音频文件"),
    ref_text: str = Form(..., description="参考音频的文本内容"),
    gen_text: str = Form(..., description="要生成的文本"),
    model: str = Form(DEFAULT_MODEL, description="模型名称")
):
    """简化版文本转语音接口，使用预加载的模型进行快速推理"""
    global f5tts_instance, model_loaded
    
    try:
        # 检查文件类型
        if not ref_audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="上传的文件不是音频格式")
        
        # 如果模型未加载，尝试加载
        if not model_loaded or f5tts_instance is None:
            print("🔄 模型未预加载，正在加载...")
            get_f5tts_instance(model, DEFAULT_CKPT_FILE, DEFAULT_VOCAB_FILE)
        
        # 保存上传的音频文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref:
            content = await ref_audio.read()
            temp_ref.write(content)
            temp_ref_path = temp_ref.name
        
        try:
            # 创建输出文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
                output_path = temp_output.name
            
            print(f"🎯 开始推理...")
            print(f"📄 参考文本: {ref_text}")
            print(f"📝 生成文本: {gen_text}")
            
            # 使用预加载的模型进行推理
            wav, sr, spec = f5tts_instance.infer(
                ref_file=temp_ref_path,
                ref_text=ref_text,
                gen_text=gen_text,
                file_wave=output_path
            )
            
            print(f"✅ 推理完成，音频已生成")
            
            return FileResponse(
                path=output_path,
                media_type="audio/wav",
                filename="generated_audio.wav"
            )
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_ref_path):
                os.unlink(temp_ref_path)
    
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

import uvicorn

if __name__ == "__main__":
    print("🚀 启动F5-TTS FastAPI服务器...")
    print("📍 默认模型: F5TTS_v1_Base")
    print("📁 模型路径: models/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")
    print("🌐 服务器地址: http://localhost:8000")
    print("📖 API文档: http://localhost:8000/docs")
    print("💡 使用 POST /preload 预加载模型")
    print("🛑 按 Ctrl+C 停止服务器")
    print("-" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # 禁用reload以避免导入问题
        log_level="info"
    )
