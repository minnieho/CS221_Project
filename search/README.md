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


### Dynamic Programming metrics

| goal dist (m)| depth | explored |  min cost  |  time (sec)| 
|:------------:|:-----:|:--------:|:----------:|:----------:|
|   25         |    5  |    453   |    5       |   0.6      |
|   30         |    6  |    916   |    6       |   1.2      |
|   40         |    8  |   2823   |    8       |   3.7      |
|   50         |   10  |   7219   |   10       |   9.8      | 
|   60         |   12  |  15223   |   12       |  20.4      |    
|   70         |   15  |  29221   |   15       |  38.6      |  
|   75         |   16  |  39475   |   16       |  54.0      |   
|   80         |   21  |  52385   |   21       |  68.1      |  
|   90         |   23  |  89742   |   23       | 118.0      |  
|   100        |    26 | 152212   |    26       |  210.6    |





  
