[33mcommit 8ea5fe197be6055f062ffce17c2fca67f13d4752[m[33m ([m[1;36mHEAD -> [m[1;32mtest-branch[m[33m)[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Tue Apr 16 15:13:57 2019 +0200

    Tidied up

[33mcommit 0c63def3cafb4884e1bcba0167b0e9fdf06b3950[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Tue Apr 16 15:05:35 2019 +0200

    Truncated measurements with updated prior mean and covariance matrix, very fast! What is the drawback?

[33mcommit 71d5643975e393067a704d72170260748d9bded5[m[33m ([m[1;31morigin/master[m[33m, [m[1;31morigin/HEAD[m[33m, [m[1;32mmaster[m[33m)[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Tue Apr 16 13:41:25 2019 +0200

    Save cache

[33mcommit 5837248170766b449ea1a5f3a721105c6de1b63c[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Mon Apr 15 15:48:58 2019 +0200

    Final commit of the day

[33mcommit ace5ad03788d6d19893b8e99b5505a850e1d3614[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Mon Apr 15 13:36:36 2019 +0200

    Save before next implementation

[33mcommit 9e83a3a3f31b40847c62c76e71aa2aebe6a3cc84[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Mon Apr 15 13:20:48 2019 +0200

    added TODOs

[33mcommit f6312993d3cff0e6ca6a334ac6ae4e2a0b1bfc03[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Mon Apr 15 13:19:31 2019 +0200

    Diagonal covariance is still wrong -> hSeq way too small in the beginning

[33mcommit fdfc7f250efcf5cf0efd3c9e1e649d8ad5104cd6[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 11 16:13:37 2019 +0200

    Sequential Bayesian Update works (except diagonal covariance calculation)
    
    TODO:
        - fix diagonal covariance calculation
        - augment to time dependent fields
            - add outer grid
            - boundary conditions
            - use of sparse commands

[33mcommit 8b79dabade663fbbd6f4747a5c59316b871d1066[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 11 15:54:49 2019 +0200

    Tidied up more

[33mcommit a6f4359fafe5b403fe95e5fa8e7eef74b62ec414[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 11 15:43:00 2019 +0200

    Tidied up (Code inspection by PyCharm), covariance still not accurate

[33mcommit aca9069367b5157cef10db82affc01dfa02a6b34[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 11 14:57:59 2019 +0200

    Sequential covariance diagonal calculation does not work yet

[33mcommit e9c6403c0b9a8129b940f993f37f330f4d27737f[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 11 14:06:55 2019 +0200

    Sequential Bayesian Regression works now

[33mcommit 9bb4f1f2f7d61c56fa6db79a4071d15d67057ead[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 11 11:55:10 2019 +0200

    Save. Results still wrong

[33mcommit 59e5c0ccd538917229cc1911ef4d54766915dcf0[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 11 09:50:50 2019 +0200

    Plot title was wrong

[33mcommit 1a02a8376fe3843527440098f9f4dbf5317f942c[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 10 16:47:16 2019 +0200

    Going home. TO DO: Check why mean and variance are wrong

[33mcommit 899502b81914d39fcf674006389a440443fec623[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 10 15:47:56 2019 +0200

    Sequential implementation is running without error, but wrong results

[33mcommit bbc79197462d7e75f7087ee0cb6dc8b0cf3a02c4[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 10 15:15:34 2019 +0200

    Switched from precision to variance plotting

[33mcommit 8a9b5725200770704e7ce297bcc4bfaa49c9d3c6[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 10 15:01:34 2019 +0200

    Added diagonal posterior covariance vector, sequential bayesian regression not debugged yet

[33mcommit 489a2d2580839fef252bf2c465dbc14644fcc907[m[33m ([m[1;32mFullLatentModel[m[33m)[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 10 14:16:00 2019 +0200

    Working Full Latent Model (not yet sequential regression)

[33mcommit 322227f9415a6a5baac9f9929b7d490991ecc202[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 10 13:58:16 2019 +0200

    tidied up

[33mcommit 96efd4b288c83b01c80c9a2293d23ecccc950ed3[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 10 12:10:11 2019 +0200

    WORKING - Fixed error in last commit

[33mcommit 03d710543a4a5f9d0d4a49d895a53c92eb90f0db[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 10 11:59:21 2019 +0200

    Regression mean with mean and covariance calculation algorithm works now. Effect of beta is hard to see, since the mean is constant (F=1)

[33mcommit 105789b1fc4d5c6cc0995f667cb36955a2b7deeb[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 10 09:48:21 2019 +0200

    Checked statement if use of matric inversion lemma leads to faster calculation if #measurements << #size of gmrfs

[33mcommit a67f78b376cca7a49820100b501de51562b7a979[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Tue Apr 9 14:42:00 2019 +0200

    Field belief not yet accurate

[33mcommit 1b1e3d87a1deba8ed1624e33a87c3ba4cd71df93[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Tue Apr 9 10:47:21 2019 +0200

    Added parameters.py, fixed wrong precision matrix

[33mcommit 5851f595c5df9519599bf1ebe2043e6ea684ec14[m
Author: bene <benedikt.mersch@gmail.com>
Date:   Mon Apr 8 19:19:01 2019 +0200

    Checked regressional implementation, still not working. Tidied up.

[33mcommit c27f2aba6e56378e5701212e44e96aaa22412766[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Mon Apr 8 14:17:42 2019 +0200

    Still working on mean regression

[33mcommit 80c94e27f48764e3ca0a040dd3664e9c1c8f29c3[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Mon Apr 8 11:28:33 2019 +0200

    Moved prior mean and precision matrix to gmrf class

[33mcommit eb6552977b82d1d8da5737257920f2427e0dbef2[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Mon Apr 8 10:47:39 2019 +0200

    renamed A to phi (according to Geist), added option of measurement mean as initial mu

[33mcommit ec11c0bee427f39b8b678532de15d586005ea789[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Sun Apr 7 21:17:39 2019 +0200

    Negative grid is now possible (by shifting xMeas and yMeas in calculation of upper left grid point; Fixed ploblem for non quadratic grid (switched order of x and y in reshaping

[33mcommit 83a9f1b4367b324202ca66cc5a1a194ce19353e7[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Sun Apr 7 16:07:18 2019 +0200

    Stepsize for state dynamics is now varied -> better exploration

[33mcommit 6ba1623086f3de5a5aa955f9e11cf6f8f26b2c5b[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Sun Apr 7 15:55:19 2019 +0200

    Plotting is now at constant speed -> needed to clear the figure every cycle. Plotting is now a method. Implementation of fast calculation Mode

[33mcommit fb32b6f588d2dd8174aeecec0a35a8695b3b7849[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Sun Apr 7 12:39:59 2019 +0200

    Worked on plotting, plotting time consumption still too large

[33mcommit ca88390aa1408b694e7f836cee94fd78cc98e01b[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 4 16:21:20 2019 +0200

    Commit before going home
    
    DONE:
            - Fixed ConDis mapping
            - implemented correct precision function
            - calculated time for GMRF update and plotted over iterations
            - implemented random walk
    
    PROBLEMS:
            - Sometimes the Phi matrix is out of bound (if outside of outer points is measured -> no neighbors)
            - non quadratic GMRF grid leads to errors (f.eg. x from 0 to 10 and y from 0 to 5)
    
    TO DO:
            - fix PROBLEMS
            - mean mu
                    - Investigate impact of one-vector / previous conditioned mean as prior mean
                    - Check what GMRF mean mu is used in Solowjow and Kreuzer
                    - Try out mu=f*theta -> mean regression
    
            - reduce plotting time
            - Enable negative grid values
            - Compare effect of Geist's improvements on calculating the update
            - augment to time dependent fields

[33mcommit 27a08b3589b069ab74aa7deb4757b6728860a3d6[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 4 16:15:44 2019 +0200

    Increased GMRF grid size -> higher increase in computation time, added print of mean of last 100 computation times

[33mcommit 33e266fb110872045e2fba4752526bde6571eae9[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 4 15:28:43 2019 +0200

    Added comments

[33mcommit 814847e1531fda661fd0fda7c037d1ebbde93991[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 4 15:17:43 2019 +0200

    Improved random walk by enforcing state constraints, visualization of walk implemented

[33mcommit 737334cd69209dae140e8462ecfe525be28242f6[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 4 14:53:08 2019 +0200

    Implemented random walk, tidied up

[33mcommit 236e1b8b605c636b0793ef1305df10eecca46b7d[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 4 14:21:01 2019 +0200

    Modified precision matrix (now basic implementation like Rue and Held,2005
    
    Tried out one-vector as pre mean of Gaussian -> much faster, less accurate
    
    TO DO:
    - convert Cont->Discrete mapping to Geist paper
    - implement random walk
    - mean mu
            - Investigate impact of one-vector / previous conditioned mean as prior mean
            - Check what GMRF mean mu is used in Solowjow and Kreuzer
            - Try out mu=f*theta -> mean regression
    
    - reduce plotting time
    - Enable negative grid values
    - Compare effect of Geist's improvements on calculating the update
    - augment to time dependent fields

[33mcommit 6239bcd18f5b5295f943883fe1aa3d3584e992de[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 4 11:50:37 2019 +0200

    added timestamps, switched of plotting (takes the most time)
    
    TO DO:
    - reduce time of sequential plotting
    - augment precision matrix
    - implement random walk

[33mcommit 7b785e422ef7bbeaddab546f5650d241ec89b100[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Thu Apr 4 11:11:03 2019 +0200

    Fixed mapping from continuous to discrete space, added band precision matrix (only covariance in x-direction)

[33mcommit 646856dfda9b89f01d1c4527863f5ca4c88e9bc4[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 3 17:55:36 2019 +0200

    Implemented GMRF as separate class
    Continuous measurements
    Mapping from continuous to discrete space
    
    Additional TO DO:
    Enable negative grid values (not possible so far, negative values don't work for calculation of neighbor index)

[33mcommit 2b2f27ccf00762648f6ba7701626cc75c5244879[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 3 14:33:01 2019 +0200

    Added variance plot

[33mcommit ca4b9b9e4564f55198c8f0b1b26e0b181b0385cb[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 3 13:49:40 2019 +0200

    First working GMRF implementation with
    - predefined ground truth on grid
    - sampling on given grid (only discrete measurements)
    - GMRF on same grid
    - random precision matrix
    - basic bayesian update
    
    TO DO:
    - plot Covariance values
    - edit precision matrix Q
    - extend to continuous space
    - extend GMRF dimensions

[33mcommit 4760144f4a6611b9fe36af0cf8ed6ea783effe12[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Wed Apr 3 13:31:52 2019 +0200

    Update improved, still not working. Problem was wrong A matrix and wrong measurement vector

[33mcommit 1eb49a2a711185ca5f1ec7774914fc6d735bb2ab[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Tue Apr 2 16:41:55 2019 +0200

    First conditioning on fake measurements, not working yet

[33mcommit 40dd50013fa7ef1a2b61ee3cba6c942cbbd0c8b8[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Tue Apr 2 14:07:11 2019 +0200

    Created first GMRF and implemented Measurements

[33mcommit 8370e07b96e9fc9e8974ceac351099d77082cb18[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Tue Apr 2 13:20:22 2019 +0200

    Added methods.py with getMeasurement method

[33mcommit 47e916d445e15a69bea98722e7c1f09094422ded[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Tue Apr 2 11:12:18 2019 +0200

    Ground truth added

[33mcommit cc69a31599d2fe980e8901291c32208963922994[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Tue Apr 2 10:18:35 2019 +0200

    Plot framework for ground truth and updating belief

[33mcommit 7782a68b9c3b574e887912657b54833243ad4a86[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Tue Apr 2 09:24:12 2019 +0200

    Changed to 2D, set up contour plot

[33mcommit 8f22379119580c75f040c3437d07bbbbf47cb58a[m
Author: Benedikt Mersch <benedikt.mersch@gmail.com>
Date:   Mon Apr 1 15:25:19 2019 +0200

    Initial commit

[33mcommit fe89a61395d12f810eddf418e857e0fd4ea1963b[m
Author: bfjs <38326482+bfjs@users.noreply.github.com>
Date:   Mon Apr 1 15:07:03 2019 +0200

    Initial commit
