# Faster than Fast

This project aims to efficiently implement ORB-FAST feature detection on embedded GPUs ([paper](https://doi.org/10.48550/arXiv.2506.07164)). Part of the code reference the [ORB-CUDA](https://github.com/chengwei920412/CUDA-ORB-local_feature) project, including the definition of the orb class, related utilities and CMake files, while our feature-point detection component demonstrates significantly improved acceleration.

> Actual performance may vary depending on CUDA architecture and memory bandwidth.
---

## Input Example

<img src="data/test.png" alt="Left Image" width="50%"><img src="data/result.png" alt="Right Image" width="50%">

&emsp;&emsp;&emsp;Input &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Output

---

## How to Compile

Use CMake to generate the project in a `build` directory:

```bash
mkdir build
cd build
cmake ..
make
```

> ⚠️ If you encounter build errors, check and adjust CUDA architecture flags in `CMakeLists.txt`.  

---

## How to Use

Run the compiled binary with the following arguments:

```bash
./FAST ../data/test.png
```

## 📂 Repository Structure
```
FasterThanFast/
│── src/
│   │── main.cpp
│   │── orb.h/.cpp 
│   │── fast_kernel.h/.cu # main GPU kernel
│   │── structures.h
│   │── utils.h
├── data/  # input images
│── CMakeLists.txt
│── LICENSE
│── README.md
```

---

## Requirements

- CUDA Toolkit ≥ 10.0
- OpenCV 4.10.0
- CMake ≥ 2.6
- GPU with compute capability 7.5+
---

## Limitations

- Gray image processing
- Limited Image size
- Fast feature detection only
---

## Troubleshooting

- **Black or zero disparity output**  
  → Likely due to incorrect CUDA architecture setting. Update `CMakeLists.txt`.

---

## What to Cite

If you use this code, please cite the following paper:

```bibtex
@article{chang2025faster,
  title={Faster than Fast: Accelerating Oriented FAST Feature Detection on Low-end Embedded GPUs},
  author={Chang, Qiong and Chen, Xinyuan and Li, Xiang and Wang, Weimin and Miyazaki, Jun},
  journal={ACM Transactions on Embedded Computing Systems},
  volume={24},
  number={3},
  pages={1--22},
  year={2025},
  publisher={ACM New York, NY}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

