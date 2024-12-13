###
Reduction:

| Size  | CPU         | GPU         | GPU 2.0     |
|-------|-------------|-------------|-------------|
| 128   | 0.000003 ns | 0.122167 ns | 0.156219 ns |
| 258   | 0.000003 ns | 0.142623 ns | 0.118813 ns |
| 1024  | 0.000008 ns | 0.114141 ns | 0.109242 ns |
---------------------------------------------------

I honeslty dont think i tested on big enough numbers for the GPU verisons to pay off, but it does seem that by 1024, the optimized GPU version begins to perform better.


###
Histogram:

| Size  | CPU         | GPU         | GPU 2.0     |
|-------|-------------|-------------|-------------|
| 128   | 0.002424 ns | 0.000129 ns | 0.000140 ns |
| 256   | 0.002060 ns | 0.000126 ns | 0.000126 ns |
| 1024  | 0.002058 ns | 0.000129 ns | 0.000173 ns |
---------------------------------------------------

For this one the optimization pays off right away between the CPU and GPU versions, but here I dont think I did enough testing on by 2 GPU versoins, non strided and strided to see the pay off of using striding. 

## Reflection Questions

2. What went well with this assignment?
3. What was difficult?
4. How would you approach differently?
5. Anything else you want me to know?


1. I think the regular gpu versions went well, I had a lot fo trouble getting the more advanced ones to work and would have to find articles/ tutorials to complete them, 
2. The more advanced GPU implementations lol
3. I would do more testing, I think my numbers were way too small to see the real impact
4. Nah <3 happy winter break!!!!!!
 