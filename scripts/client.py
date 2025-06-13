import pygame
from PIL import Image
import numpy as np
import socket
import pickle
from collections import deque

import pygame.colordict

from protocol import *

pygame.init()
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height))

pygame.display.set_caption("Periodic Image Display")

def pilImageToSurface(pilImage):
    return pygame.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

clock = pygame.time.Clock()

from socket_client import request_sensor_data,send_action_message

run = True
data = None

lastx = 0 
lasty = 0

points = deque(np.zeros((0,2),dtype = float),maxlen=3000)
x,y,o = 0,0,0
offset = np.array([900,300])

init_pos = None
init_rot = None
from scipy.spatial.transform import Rotation

while run:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                x = 1
            if event.key == pygame.K_s:
                x = -0.5
            if event.key == pygame.K_a:               
                o = 0.6
            if event.key == pygame.K_d:
                o = -0.6
            if event.key == pygame.K_SPACE:
                x,y,o = 0,0,0
            send_action_message(VelMessage(x,y,o))

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                x = 0
            if event.key == pygame.K_s:
                x = 0
            if event.key == pygame.K_a:               
                o = 0
            if event.key == pygame.K_d:
                o = 0
            if event.key == pygame.K_SPACE:
                x,y,o = 0,0,0
            send_action_message(VelMessage(x,y,o))
        
        if event.type ==  pygame.MOUSEBUTTONDOWN:
            x,y = (np.array(pygame.mouse.get_pos())-offset-np.array([screen.get_rect().x,screen.get_rect().y]))*np.array([1,-1])
            wps = np.vstack((points[-1]-init_pos,points[-1]+np.array([x,y])*10-init_pos))

            print(wps)
            translations = np.hstack((wps,np.ones((len(wps),1))*0.2)) @ init_rot

            waypoints = WaypointMessage()
            waypoints.x = translations[:,0]
            waypoints.z = translations[:,1]
            # send_action_message(waypoints)

    # send_action_message(VelMessage(1,0,0))
    # send_action_message(VelMessage(1,0,0))

    try:
        data = request_sensor_data()

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


        points.append([p['x'],p['y']])
        if init_pos is None:
            init_pos = np.array(points[-1])
            init_rot = Rotation.from_quat([o['x'],o['y'],o['z'],o['w']]).as_matrix()

        p = np.array(points)*np.array([[1,-1]])

        server_timestamp_ns = data.get("timestamp_server_ns")
        screen.fill(0)


        if len(points)>1:
            for i in range(1,len(points)):
                pygame.draw.line(screen, pygame.Color('aqua'),(p[i-1]-p[-1])*40+offset, (p[i]-p[-1])*40+offset) 

        pygameSurface = pilImageToSurface(Image.fromarray(rgb_image,mode='RGB'))
        screen.blit(pygameSurface, pygameSurface.get_rect(center = (250, 250)))
        pygame.display.flip()