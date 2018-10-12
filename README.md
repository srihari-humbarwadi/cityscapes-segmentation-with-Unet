# Readme
The model was trained on cityscape-dataset with classes merged into 8 main classes.
The model was trained for 300k iterations with lr=0.001, and no wd.
Results are expected to improve drastically after hyperparameter tuning
You can download the pretrained weights from here
[Pretrained weights](https://drive.google.com/file/d/1RaRn9eI40ZDXAX-ajnCT0oWJclR8w9RF/view?usp=sharing)
## List of classes
  - void
  - flat
  - construction
  - object
  - nature
  - sky
  - human
  - vehicle

# Results
The results have only 3 classes marked on them. 
 - human (Red)
 - construction (Green)
 - vehicle (Blue)
 
## Video results
[Part 1](https://youtu.be/D68uL3aEu5o)
[Part 2](https://youtu.be/YwalCv13E0Q)

## Image results
![image 1](outputs/results/1264.png)
![image 1](outputs/results/1356.png)
![image 1](outputs/results/1351.png)
![image 1](outputs/results/1743.png)
  
## To-do
- [ ] Hyperparameter tuning
- [ ] Extend to cover all the original classes
- [X] Try with Conv2DTranspose
