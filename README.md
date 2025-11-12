# distributed-ml-training

(WIP)

This repository is my learning journey for **scaling deep learning training beyond one GPU**.

The goal is to learn step-by-step:

- how multi-GPU training actually works
- how datasets & models must be managed in distributed context
- how to evaluate & checkpoint correctly across ranks
- how to measure scaling performance
- how to move from DDP → FSDP → ZeRO when models get large


This repo is structured as small incremental chapters — each chapter has:
- a specific scope & learning goal
- runnable code examples
- short notes (what I learned)

---

## Chapter List

| Folder | Title | Instructional Goal |
|---|---|---|
| **01_single_gpu_to_ddp** | From Single GPU to Basic DDP | understand *what changes* between normal training and multi-GPU |
