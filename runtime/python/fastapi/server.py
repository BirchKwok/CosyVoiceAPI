# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from pathlib import Path
import sys
import argparse
import logging
from typing import Optional
import io
import wave

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    """生成WAV格式的音频数据"""
    # 收集所有音频数据
    audio_chunks = []
    for i in model_output:
        audio_data = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16)
        audio_chunks.append(audio_data)
    
    # 将所有音频数据合并
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        
        # 创建WAV文件，使用模型的采样率
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(cosyvoice.sample_rate)  # 使用模型的采样率
            wav_file.writeframes(full_audio.tobytes())
        
        wav_buffer.seek(0)
        yield wav_buffer.read()


@app.get("/hello")
async def hello():
    """
    返回API服务的问候消息
    """
    return {"message": "你好！我是CosyVoice2 API服务。很高兴为您服务！", 
            "version": "2.0", 
            "model_type": "CosyVoice2",
            "sample_rate": cosyvoice.sample_rate,
            "available_endpoints": [
                "/inference_sft",
                "/inference_zero_shot", 
                "/inference_cross_lingual",
                "/inference_instruct",  # 注意：CosyVoice2不支持此方法
                "/inference_instruct2",
                "/tts/clone",
                "/hello"
            ],
            "notes": {
                "inference_instruct": "CosyVoice2不支持此方法，请使用inference_instruct2",
                "audio_format": "16kHz输入，24kHz输出（CosyVoice2默认）"
            }}


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    """SFT模式语音合成"""
    try:
        logging.info(f"SFT合成请求: 文本='{tts_text}', 说话人ID='{spk_id}'")
        model_output = cosyvoice.inference_sft(tts_text, spk_id)
        return StreamingResponse(generate_data(model_output), media_type="audio/wav")
    except Exception as e:
        logging.error(f"SFT合成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SFT合成失败: {str(e)}")


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    """零样本语音合成"""
    try:
        logging.info(f"零样本合成请求: 文本='{tts_text}', 参考文本='{prompt_text}'")
        # 加载参考音频，16kHz输入会被自动重采样
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
        return StreamingResponse(generate_data(model_output), media_type="audio/wav")
    except Exception as e:
        logging.error(f"零样本合成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"零样本合成失败: {str(e)}")


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    """跨语言语音合成"""
    try:
        logging.info(f"跨语言合成请求: 文本='{tts_text}'")
        # 加载参考音频，16kHz输入会被自动重采样
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
        return StreamingResponse(generate_data(model_output), media_type="audio/wav")
    except Exception as e:
        logging.error(f"跨语言合成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"跨语言合成失败: {str(e)}")


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    """
    指令式合成接口
    注意：CosyVoice2不支持inference_instruct，需要提供参考音频使用inference_instruct2
    """
    try:
        # CosyVoice2不支持inference_instruct方法
        if hasattr(cosyvoice, 'inference_instruct'):
            model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
        else:
            # CosyVoice2不支持这个方法，返回错误提示
            raise HTTPException(
                status_code=400, 
                detail="CosyVoice2不支持inference_instruct方法，请使用inference_instruct2并提供参考音频"
            )
        return StreamingResponse(generate_data(model_output))
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"指令式合成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"指令式合成失败: {str(e)}")


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    """指令式合成2（CosyVoice2支持的版本）"""
    try:
        logging.info(f"指令式合成2请求: 文本='{tts_text}', 指令='{instruct_text}'")
        # 加载参考音频，16kHz输入会被自动重采样
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
        return StreamingResponse(generate_data(model_output), media_type="audio/wav")
    except Exception as e:
        logging.error(f"指令式合成2失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"指令式合成2失败: {str(e)}")


@app.get("/tts/clone")
@app.post("/tts/clone")
async def tts_clone(
    text: str = Form(..., description="要合成的文本"),
    prompt_audio: UploadFile = File(..., description="参考音频文件，用于声音克隆"),
    prompt_text: Optional[str] = Form(None, description="参考音频的文本内容（可选）"),
    cross_lingual: bool = Form(False, description="是否进行跨语言合成")
):
    """
    声音克隆接口 - 使用上传的音频文件克隆声音特征来合成新的语音
    
    参数:
        text: 要合成的文本
        prompt_audio: 参考音频文件，用于提取声音特征
        prompt_text: 参考音频对应的文本（可选，用于零样本合成）
        cross_lingual: 是否使用跨语言模式（默认False）
    
    返回:
        音频流数据
    """
    try:
        # 加载参考音频，采样率16kHz
        prompt_speech_16k = load_wav(prompt_audio.file, 16000)
        
        if cross_lingual:
            # 使用跨语言合成模式
            logging.info(f"使用跨语言模式合成语音: {text}")
            model_output = cosyvoice.inference_cross_lingual(text, prompt_speech_16k)
        else:
            # 使用零样本合成模式
            if prompt_text is None or len(prompt_text.strip()) == 0:
                # 如果没有提供参考文本，使用跨语言模式作为fallback
                logging.info(f"未提供参考文本，使用跨语言模式: {text}")
                model_output = cosyvoice.inference_cross_lingual(text, prompt_speech_16k)
            else:
                logging.info(f"使用零样本模式合成语音: {text}, 参考文本: {prompt_text}")
                model_output = cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k)
        
        return StreamingResponse(generate_data(model_output), 
                               media_type="audio/wav",
                               headers={"Content-Disposition": "attachment; filename=cloned_voice.wav"})
                               
    except Exception as e:
        logging.error(f"声音克隆过程中出现错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"声音克隆失败: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8080)
    # 文件当前路径
    model_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent / "pretrained_models" / "CosyVoice2-0.5B"
    parser.add_argument('--model_dir',
                        type=str,
                        default=model_dir.as_posix(),
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    
    # 强制使用CosyVoice2
    try:
        cosyvoice = CosyVoice2(args.model_dir)
        logging.info(f"成功加载CosyVoice2模型，采样率: {cosyvoice.sample_rate}Hz")
    except Exception as e:
        logging.error(f"加载CosyVoice2模型失败: {str(e)}")
        raise TypeError(f'无法加载CosyVoice2模型: {str(e)}')
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
