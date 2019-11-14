# CS221_Project  

**Project install**  

```bash
git clone https://github.com/PhilippeW83440/CS221_Project.git
cd CS221_Project
pip install -r requirements.txt
cd gym-act
pip install -e .
```
Then check you can run the notebook Test_Setup.ipynb without error

### Using a virtual env

We recommend using python3. You may find convenient to use a virtual env.

```bash
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with ```deactivate```.

### Uniform Cost Search metrics

During the tree search we use a Time Step of 250 ms.  
Which means with a depth of 20 we can explore over a 5 seconds time horizon.
The scene is setup with cars including ego vehicle driving around 20 m/s.
So in 5 seconds we typically cover distances of 100 meters.

| goal dist (m)| depth | explored |  min cost  |  time (sec)| 
|:------------:|:-----:|:--------:|:----------:|:----------:|
|   25         |    5  |    246   |    5       |   0.3      |
|   30         |    6  |    494   |    6       |   0.6      |
|   40         |    8  |   1516   |    8       |   2.0      |
|   50         |   10  |   3649   |   10       |   5.0      | 
|   60         |   12  |   7509   |   12       |   9.9      |    
|   70         |   15  |  15231   |   15       |  20.0      |   
|   75         |   16  |  17977   |   16       |  24.2      |   
|   80         |   21  |  27626   |   21       |  37.1      |  
|   90         |   23  |  30829   |   23       |  39.9      |  
|  100         |   26  |  36834   |   26       |  49.6      |  




  
