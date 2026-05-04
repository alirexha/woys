"""RVC-only voice changer manager (vcclient-cachy fork).

Single-engine dispatch: every non-RVC arm in the upstream loader/runner has
been collapsed. The loadModel / generateVoiceChanger paths now only handle
voiceChangerType == "RVC".
"""
import json
import os
import re
import shutil
import sys
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from const import STORED_SETTING_FILE, UPLOAD_DIR, StaticSlot
from downloader.SampleDownloader import downloadSample, getSampleInfos
from mods.log_control import VoiceChangaerLogger
from voice_changer.Local.ServerDevice import ServerDevice, ServerDeviceCallbacks
from voice_changer.ModelSlotManager import ModelSlotManager
from voice_changer.RVC.RVCModelMerger import RVCModelMerger
from voice_changer.utils.LoadModelParams import LoadModelParamFile, LoadModelParams
from voice_changer.utils.ModelMerger import MergeElement, ModelMergerRequest
from voice_changer.utils.VoiceChangerModel import AudioInOut
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from voice_changer.VoiceChangerV2 import VoiceChangerV2

logger = VoiceChangaerLogger.get_instance().getLogger()


@dataclass()
class GPUInfo:
    id: int
    name: str
    memory: int


@dataclass()
class VoiceChangerManagerSettings:
    modelSlotIndex: int | StaticSlot = -1
    passThrough: bool = False
    boolData: list[str] = field(default_factory=lambda: ["passThrough"])
    intData: list[str] = field(default_factory=lambda: ["modelSlotIndex"])


class VoiceChangerManager(ServerDeviceCallbacks):
    _instance = None

    # ServerDeviceCallbacks ----------------------------------------------------

    def on_request(self, unpackedData: AudioInOut):
        return self.changeVoice(unpackedData)

    def emitTo(self, performance: list[float]):
        self.emitToFunc(performance)

    def get_processing_sampling_rate(self):
        return self.voiceChanger.get_processing_sampling_rate()

    def setInputSamplingRate(self, sr: int):
        self.voiceChanger.setInputSampleRate(sr)

    def setOutputSamplingRate(self, sr: int):
        self.voiceChanger.setOutputSampleRate(sr)

    # VoiceChangerManager ------------------------------------------------------

    def __init__(self, params: VoiceChangerParams):
        logger.info("[Voice Changer] VoiceChangerManager initializing...")
        self.params = params
        self.voiceChanger: VoiceChangerV2 | None = None
        self.settings = VoiceChangerManagerSettings()

        self.modelSlotManager = ModelSlotManager.get_instance(self.params.model_dir)
        self.gpus: list[GPUInfo] = self._get_gpuInfos()

        self.serverDevice = ServerDevice(self)
        thread = threading.Thread(target=self.serverDevice.start, args=())
        thread.start()

        self.stored_setting: dict[str, str | int | float] = {}
        if os.path.exists(STORED_SETTING_FILE):
            self.stored_setting = json.load(open(STORED_SETTING_FILE, "r", encoding="utf-8"))
        if "modelSlotIndex" in self.stored_setting:
            self.update_settings("modelSlotIndex", self.stored_setting["modelSlotIndex"])
        if "gpu" not in self.stored_setting:
            self.update_settings("gpu", 0)
        logger.info("[Voice Changer] VoiceChangerManager initializing... done.")

    def store_setting(self, key: str, val: str | int | float):
        saveItemForServerDevice = [
            "enableServerAudio",
            "serverAudioSampleRate",
            "serverInputDeviceId",
            "serverOutputDeviceId",
            "serverMonitorDeviceId",
            "serverReadChunkSize",
            "serverInputAudioGain",
            "serverOutputAudioGain",
        ]
        saveItemForVoiceChanger = ["crossFadeOffsetRate", "crossFadeEndRate", "crossFadeOverlapSize"]
        saveItemForVoiceChangerManager = ["modelSlotIndex"]
        saveItemForRVC = ["extraConvertSize", "gpu", "silentThreshold"]
        saveItemForAllVoiceChanger = ["f0Detector"]

        saveItem: list[str] = []
        saveItem.extend(saveItemForServerDevice)
        saveItem.extend(saveItemForVoiceChanger)
        saveItem.extend(saveItemForVoiceChangerManager)
        saveItem.extend(saveItemForRVC)
        saveItem.extend(saveItemForAllVoiceChanger)
        if key in saveItem:
            self.stored_setting[key] = val
            json.dump(self.stored_setting, open(STORED_SETTING_FILE, "w"))

    def _get_gpuInfos(self):
        devCount = torch.cuda.device_count()
        gpus = []
        for id in range(devCount):
            name = torch.cuda.get_device_name(id)
            memory = torch.cuda.get_device_properties(id).total_memory
            gpus.append({"id": id, "name": name, "memory": memory})
        return gpus

    @classmethod
    def get_instance(cls, params: VoiceChangerParams):
        if cls._instance is None:
            cls._instance = cls(params)
        return cls._instance

    def loadModel(self, params: LoadModelParams):
        if params.isSampleMode:
            logger.info(f"[Voice Changer] sample download...., {params}")
            downloadSample(
                self.params.sample_mode,
                params.sampleId,
                self.params.model_dir,
                params.slot,
                params.params,
            )
            self.modelSlotManager.getAllSlotInfo(reload=True)
            return {"status": "OK"}

        # Upload path: copy files into the slot dir.
        slotDir = os.path.join(self.params.model_dir, str(params.slot))
        if os.path.isdir(slotDir):
            shutil.rmtree(slotDir)

        for file in params.files:
            logger.info(f"FILE: {file}")
            srcPath = os.path.join(UPLOAD_DIR, file.dir, file.name)
            dstDir = os.path.join(self.params.model_dir, str(params.slot), file.dir)
            dstPath = os.path.join(dstDir, file.name)
            os.makedirs(dstDir, exist_ok=True)
            logger.info(f"move to {srcPath} -> {dstPath}")
            shutil.move(srcPath, dstPath)
            file.name = os.path.basename(dstPath)

        if params.voiceChangerType == "RVC":
            # Late import: model_dir/etc must be set in params first.
            from voice_changer.RVC.RVCModelSlotGenerator import RVCModelSlotGenerator

            slotInfo = RVCModelSlotGenerator.loadModel(params)
            self.modelSlotManager.save_model_slot(params.slot, slotInfo)
        else:
            logger.warning(
                f"[Voice Changer] vcclient-cachy is RVC-only; ignoring "
                f"unsupported voiceChangerType={params.voiceChangerType!r}"
            )

        logger.info(f"params, {params}")

    def get_info(self):
        data = asdict(self.settings)
        data["gpus"] = self.gpus
        data["modelSlots"] = self.modelSlotManager.getAllSlotInfo(reload=True)
        data["sampleModels"] = getSampleInfos(self.params.sample_mode)
        data["python"] = sys.version
        data["voiceChangerParams"] = self.params
        data["status"] = "OK"

        info = self.serverDevice.get_info()
        data.update(info)

        if self.voiceChanger is not None:
            info = self.voiceChanger.get_info()
            data.update(info)
        return data

    def get_performance(self):
        if hasattr(self, "voiceChanger") and self.voiceChanger is not None:
            return self.voiceChanger.get_performance()
        return {"status": "ERROR", "msg": "no model loaded"}

    def generateVoiceChanger(self, val: int | StaticSlot):
        slotInfo = self.modelSlotManager.get_slot_info(val)
        if slotInfo is None:
            logger.info(f"[Voice Changer] model slot is not found {val}")
            return
        if slotInfo.voiceChangerType == "RVC":
            logger.info("................RVC")
            from voice_changer.RVC.RVCr2 import RVCr2

            self.voiceChangerModel = RVCr2(self.params, slotInfo)
            self.voiceChanger = VoiceChangerV2(self.params)
            self.voiceChanger.setModel(self.voiceChangerModel)
        else:
            logger.info(
                f"[Voice Changer] vcclient-cachy is RVC-only; "
                f"unsupported model type {slotInfo.voiceChangerType!r}"
            )
            if hasattr(self, "voiceChangerModel"):
                del self.voiceChangerModel
            return

    def update_settings(self, key: str, val: str | int | float | bool):
        self.store_setting(key, val)

        if key in self.settings.boolData:
            newVal: Any = val
            if val == "true":
                newVal = True
            elif val == "false":
                newVal = False
            setattr(self.settings, key, newVal)
        elif key in self.settings.intData:
            if key == "modelSlotIndex":
                try:
                    newVal = int(val) % 1000
                except (TypeError, ValueError):
                    newVal = re.sub(r"^\d+", "", str(val))
                logger.info(
                    f"[Voice Changer] model slot is changed "
                    f"{self.settings.modelSlotIndex} -> {newVal}"
                )
                self.generateVoiceChanger(newVal)
                for k, v in self.stored_setting.items():
                    if k != "modelSlotIndex":
                        self.update_settings(k, v)
            else:
                newVal = int(val)
            setattr(self.settings, key, newVal)

        self.serverDevice.update_settings(key, val)
        if self.voiceChanger is not None:
            self.voiceChanger.update_settings(key, val)
        return self.get_info()

    def changeVoice(self, receivedData: AudioInOut):
        if self.settings.passThrough is True:
            return receivedData, []
        if self.voiceChanger is not None:
            return self.voiceChanger.on_request(receivedData)
        logger.info("Voice Change is not loaded. Did you load a correct model?")
        return np.zeros(1).astype(np.int16), []

    def export2onnx(self):
        return self.voiceChanger.export2onnx()

    def merge_models(self, request: str):
        req_dict = json.loads(request)
        req = ModelMergerRequest(**req_dict)
        req.files = [MergeElement(**f) for f in req.files]
        # Beatrice-JVS slot is gone; no -2 adjustment.
        slot = len(self.modelSlotManager.getAllSlotInfo()) - 1
        if req.voiceChangerType == "RVC":
            merged = RVCModelMerger.merge_models(self.params, req, slot)
            loadParam = LoadModelParams(
                voiceChangerType="RVC",
                slot=slot,
                isSampleMode=False,
                sampleId="",
                files=[
                    LoadModelParamFile(name=os.path.basename(merged), kind="rvcModel", dir="")
                ],
                params={},
            )
            self.loadModel(loadParam)
        return self.get_info()

    def setEmitTo(self, emitTo: Callable[[Any], None]):
        self.emitToFunc = emitTo

    def update_model_default(self):
        current_settings = self.voiceChangerModel.get_model_current()
        for current_setting in current_settings:
            current_setting["slot"] = self.settings.modelSlotIndex
            self.modelSlotManager.update_model_info(json.dumps(current_setting))
        return self.get_info()

    def update_model_info(self, newData: str):
        self.modelSlotManager.update_model_info(newData)
        return self.get_info()

    def upload_model_assets(self, params: str):
        self.modelSlotManager.store_model_assets(params)
        return self.get_info()
