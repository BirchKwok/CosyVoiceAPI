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
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/hello")
async def hello():
    """
    返回API服务的问候消息
    """
    return {"message": "你好！我是CosyVoice API服务。很高兴为您服务！", 
            "version": "1.0", 
            "available_endpoints": [
                "/inference_sft",
                "/inference_zero_shot", 
                "/inference_cross_lingual",
                "/inference_instruct",
                "/inference_instruct2",
                "/voice_clone",
                "/hello"
            ]}


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


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
    try:
        cosyvoice = CosyVoice2(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)
