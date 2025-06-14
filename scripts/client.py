import pygame
from PIL import Image
import numpy as np
import socket
import pickle
from collections import deque

import pygame.colordict

from protocol import *
from argparse import ArgumentParser

#preprogrammed waypoints to execute by pressing enter.
WAYPOINTS = np.array([
[0,0],[0.5,0.2],[1,0],[1.5,-0.2],[2,0]
])

parser = ArgumentParser()
parser.add_argument("--host",type=str,default='localhost')
args = parser.parse_args()
pygame.init()
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height))

pygame.display.set_caption("Periodic Image Display")

def pilImageToSurface(pilImage):
    return pygame.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

clock = pygame.time.Clock()

from socket_client import request_sensor_data,send_action_message,request_planner_state

run = True
data = None

lastx = 0 
lasty = 0

points = deque(np.zeros((0,2),dtype = float),maxlen=3000)
vx,vy,omg = 0,0,0
offset = np.array([900,300])

init_T = None
curr_T = None

from scipy.spatial.transform import Rotation
from copy import deepcopy



waypointmsg = WaypointMessage()

waypointmsg.x = WAYPOINTS[:,0]
waypointmsg.z = WAYPOINTS[:,1]

while run:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            # print(Rotation.from_matrix(curr_T[:3,:3]).as_quat())
            if event.key == pygame.K_w:
                vx = 0.5
            if event.key == pygame.K_s:
                vx = -0.5
            if event.key == pygame.K_a:               
                omg = 0.5
            if event.key == pygame.K_d:
                omg = -0.5
            if event.key == pygame.K_q:
                vy = -0.5
            if event.key == pygame.K_e:
                vy = 0.5
            if event.key == pygame.K_SPACE:
                vx,vy,omg = 0,0,0
            if event.key == pygame.K_RETURN:
                print("executing preprgrammed trajectory:")
                translations = np.hstack((WAYPOINTS,np.ones((len(WAYPOINTS),1))*0.2,np.ones((len(WAYPOINTS),1)))) @  curr_T.T# @ np.linalg.inv(init_T).T 
                
                waypointmsg.x = translations[:,0]
                waypointmsg.z = translations[:,1] #invert it because z is positive right, but y is positive left.

                send_action_message(waypointmsg, host=args.host)
                continue
                
            send_action_message(VelMessage(vx,vy,omg), host=args.host)
            
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                vx = 0
            if event.key == pygame.K_s:
                vx = 0
            if event.key == pygame.K_a:               
                omg = 0
            if event.key == pygame.K_d:
                omg = 0
            if event.key == pygame.K_q:
                vy = 0
            if event.key == pygame.K_e:
                vy = 0
            if event.key == pygame.K_SPACE:
                vx,vy,omg = 0,0,0
            send_action_message(VelMessage(vx,vy,omg),args.host)
        
        # if event.type ==  pygame.MOUSEBUTTONDOWN:
        #     x,y = (np.array(pygame.mouse.get_pos())-offset-np.array([screen.get_rect().x,screen.get_rect().y]))*np.array([1,-1])
        #     wps = np.vstack((points[-1]-init_pos,points[-1]+np.array([x,y])*10-init_pos))

        #     print(wps)
        #     translations = np.hstack((wps,np.ones((len(wps),1))*0.2)) @ init_rot

        #     waypoints = WaypointMessage()
        #     waypoints.x = translations[:,0]
        #     waypoints.z = translations[:,1]
            # send_action_message(waypoints,args.host)


    try:
        data = request_sensor_data(args.host)
        my_dict = request_planner_state(args.host)
        decimal_places = 3
        rounded_dict = {k: round(v, decimal_places) if isinstance(v, float) else v for k, v in my_dict.items()}
        # print(rounded_dict)

    except socket.timeout:
            print(f"Socket timeout during operation with {SERVER_HOST}:{SERVER_PORT}")
    except socket.error as e:
        print(f"Socket error: {e}")
    except pickle.UnpicklingError as e:
        print(f"Pickle error: {e}. Received data might be corrupt or not a pickle.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if 'client_socket' in locals():
            client_socket.close()

    if data and data.get("success", False):
        rgb_image = data.get("rgb_image")
        depth_image = data.get("depth_image")
        pose = data.get("pose")
        p = pose['pose']['position']
        o = pose['pose']['orientation']

        curr_T = np.eye(4)
        curr_T[:3,:3] = Rotation.from_quat([o['x'],o['y'],o['z'],o['w']]).as_matrix()
        curr_T[:3,3] = np.array([p['x'],p['y'],p['z']])
        curr_T = deepcopy(curr_T)
        if init_T is None:
            init_T = deepcopy(curr_T)
            print(init_T)

        points.append([p['x'],p['y']])

        p = np.array(points)*np.array([[1,-1]])

        server_timestamp_ns = data.get("timestamp_server_ns")
        screen.fill(0)


        if len(points)>1:
            for i in range(1,len(points)):
                pygame.draw.line(screen, pygame.Color('green'),(p[i-1]-p[-1])*40+offset, (p[i]-p[-1])*40+offset) 

        pygameSurface = pilImageToSurface(Image.fromarray(rgb_image,mode='RGB'))
        screen.blit(pygameSurface, pygameSurface.get_rect(center = (250, 250)))
        font = pygame.font.Font('freesansbold.ttf', 32)

        # create a text surface object,
        # on which text is drawn on it.
        green = (0, 255, 0)
        blue = (0, 0, 128)
        text = font.render(str(rounded_dict), True, green, blue)
        screen.blit(text,(10,600))
        pygame.display.flip()