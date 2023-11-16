# 3DrosEnv

How to setup:
```
docker compose up
```

How does this work?

1. We have the first image. This topic uses the entrypoint.sh file.
   - This is responsible for setting up the gazebo and runs the 1v1 competition.
2. We have the second image - This image is uses the entrypoint2.sh and newpub.py file.
   - This is responsible for initializing the first "shooter"
   - The first shooter will move around randomly
3. We have the third image - this image uses the entrypoint3.sh file and pub_for_shooting.py file.
   - This is responsible for initializing the shooter topic
   - This will shoot bullets randomly

4. Then, the next image (interaction2) does the same thing as the second image, but for the blue shooter
   - it uses the files entrypoint4.sh and newpub2.py
5. Then it has the 5th image (rviz_setup2). This is a copy of the third image, but for the blue shooter
   - It uses the files entrypoint5.sh and pub_for_shooting2.py
6. Then we have the two health display images - these display the referee information for both (health, bullets, etc)
   
