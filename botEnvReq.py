import RPi.GPIO as GPIO
import time
import requests 

class Bot: 

    #Has methods for movement, setting servo angle, and getting obstacle distance
    def __init__(self): 
        self.setServoAngle (90) 
        print("Bot Created.") 

    def setServoAngle(self, angle):  #angle in degrees 

        GPIO.setmode(GPIO.BOARD) 
        GPIO.setup(12,GPIO.OUT)
        p = GPIO.PWM(12,50)
        p.start(0)
        
        dutyCycle = angle/18 + 2  
        p.ChangeDutyCycle(dutyCycle) 
        time.sleep(0.1)  
        p.stop()
        GPIO.cleanup()

    def getObstacleDist(self): 

        GPIO.setmode(GPIO.BCM)
        #set GPIO Pins
        GPIO_TRIGGER = 13
        GPIO_ECHO = 19
 
        #set GPIO direction (IN / OUT)
        GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
        GPIO.setup(GPIO_ECHO, GPIO.IN)
 
        GPIO.output(GPIO_TRIGGER, True)
        # set Trigger after 0.01ms to LOW
        time.sleep(0.00001)
        GPIO.output(GPIO_TRIGGER, False)
 
        StartTime = time.time()
        StopTime = time.time()
 
        # save StartTime
        while GPIO.input(GPIO_ECHO) == 0:
            StartTime = time.time()   
        # save time of arrival
        while GPIO.input(GPIO_ECHO) == 1:
            StopTime = time.time()
 
        TimeElapsed = StopTime - StartTime
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distance = (TimeElapsed * 34300) / 2 
        GPIO.cleanup() 
 
        return distance 
        
    def moveInit(self): 
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(17, GPIO.OUT)
        GPIO.setup(22, GPIO.OUT)
        GPIO.setup(23, GPIO.OUT)
        GPIO.setup(24, GPIO.OUT) 

    def forward(self, sec):
        self.moveInit() 
        GPIO.output(22, True)
        GPIO.output(17, False)
        GPIO.output(23, True) 
        GPIO.output(24, False)
        time.sleep(sec)
        GPIO.cleanup()


    def left(self, sec):
        self.moveInit() 
        GPIO.output(22, True)
        GPIO.output(17, False)
        GPIO.output(23, False) 
        GPIO.output(24, False)
        time.sleep(sec)
        GPIO.cleanup()
        
        
    def right(self, sec):
        self.moveInit() 
        GPIO.output(22, False)
        GPIO.output(17, False)
        GPIO.output(23, True) 
        GPIO.output(24, False)
        time.sleep(sec) 
        GPIO.cleanup() 


    def reverse(self, sec):
        self.moveInit() 
        GPIO.output(22, False)
        GPIO.output(17, True)
        GPIO.output(23, False) 
        GPIO.output(24, True)
        time.sleep(sec)
        GPIO.cleanup()

class BotControl: 

    def __init__(self): 
        self.bot = Bot() 
        self.startTime = time.time() 
        self.state = None 
        # self.facing = Think about it 
        self.facing = "Front"     
        print ("Bot Ready.")

        self.API_ENDPOINT = "" #server ip 
        self.API_KEY = "" #Authentication         

    def observe (self): 

        obs = [] 
        for i in range (0,180,45): 
            self.bot.setServoAngle(i) 
            obs.append(self.bot.getObstacleDist())   

        return obs 

    def comm(self): #Post requests 

        env = dict() #Dictionary containing readings, terminal, and reward  
        action = 0 

        while True:             
            
            observationSpace = self.observe() 
            terminal = False 
            reward = self.envStep(action)  

            for x in observationSpace: 
                if x < 50: 
                    terminal = True 
                    break 

            if time.time() - self.startTime > 200: 
                terminal = True 
                self.startTime = time.time() 
            
            env["o1"] = observationSpace[0] 
            env["o2"] = observationSpace[1] 
            env["o3"] = observationSpace[2] 
            env["o4"] = observationSpace[3] 
            env["terminal"] = terminal
            env["reward"] = reward  

            r = requests.post(url = self.API_ENDPOINT, data = env)  
            action = r.text   #Update action 
            print ("Received command", action)  
            time.sleep(0.5) 
            
        
    #0 - front 
    #1 - reverse 
    #2 - left 
    #3 - right 
    def getReward (self, action):  

        if self.facing == "Front":
            if action == 0: 
                return 1
            elif action == 1: 
                return -1 
            else: 
                return 0 
        
        elif self.facing == "Back": 
            if action == 0: 
                return -1 
            elif action == 1: 
                return 1
            else: 
                return 0 
        
        elif self.facing == "Left": 
            if action == 2: 
                return -1 
            elif action == 3: 
                return 1 
            else: 
                return 0 

        else: 
            if action == 2: 
                return 1 
            elif action == 3: 
                return -1 
            else: 
                return 0                 

    #Observation space is ultrasonic reading at angles of 10,20,...,180.
    def envStep(self, action): #takes action and returns associated reward
        
        reward = self.getReward(action)  
        
        if action == 0: 
            self.bot.forward(0.5) 
        elif action == 1: 
            self.bot.reverse(0.5) 

        elif action == 2: 
            self.bot.left(1.45) 
            if self.facing == "Front": 
                self.facing = "Left" 
            elif self.facing == "Left": 
                self.facing = "Back" 
            elif self.facing == "Back": 
                self.facing = "Right" 
            else: 
                self.facing = "Front"  
        
        else: 
            self.bot.right(1.45)
            if self.facing == "Front": 
                self.facing = "Right"  
            elif self.facing == "Right": 
                self.facing = "Back"   
            elif self.facing == "Back": 
                self.facing = "Left"  
            else: 
                self.facing = "Front"  
        
        print ("Done action.", flush=True) 
        return reward 

try: 
    botmove = BotControl() 
    botmove.comm()  
except KeyboardInterrupt: 
    print ("Stopping bot.")  
    GPIO.cleanup() 