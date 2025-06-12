import pygame
from PIL import Image
import numpy as np
import socket
import pickle

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
while run:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.K_w:
            send_action_message(VelMessage(1,0,0))
        if event.type == pygame.K_a:
            send_action_message(VelMessage(0,0,-0.3))
        if event.type == pygame.K_d:
            send_action_message(VelMessage(0,0,0.3))

        if event.type == pygame.K_SPACE:
            send_action_message(VelMessage(0,0,0))
 
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
        server_timestamp_ns = data.get("timestamp_server_ns")
        screen.fill(0)
        pygameSurface = pilImageToSurface(Image.fromarray(rgb_image,mode='RGB'))
        screen.blit(pygameSurface, pygameSurface.get_rect(center = (250, 250)))
        pygame.display.flip()