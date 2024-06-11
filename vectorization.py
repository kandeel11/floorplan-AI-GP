from PIL import Image, ImageDraw
import numpy as np
import shutil
import utils
import time
import os
import removebackground
data_size = 256

class WallAbstract():
    def __init__(self, input_map):  
        self.boundary_map = input_map[:,:,0]    
        self.interior_wall_map = input_map[:,:,1] 
        self.label_map = input_map[:,:,2] 
        self.inside_map = input_map[:,:,3] 
        self.exterior_boundary = []
        self.interior_walls = []

        self.house_min_h = data_size
        self.house_max_h = 0
        self.house_min_w = data_size
        self.house_max_w = 0
        for h in range(data_size):  
            for w in range(data_size):
                if self.boundary_map[h, w] > 0:
                    self.house_min_h = h if(self.house_min_h > h) else self.house_min_h
                    self.house_max_h = h if(self.house_max_h < h) else self.house_max_h
                    self.house_min_w = w if(self.house_min_w > w) else self.house_min_w
                    self.house_max_w = w if(self.house_max_w < w) else self.house_max_w
        self.house_min_h -= 10
        self.house_max_h += 10
        self.house_min_w -= 10
        self.house_max_w += 10

    def scan_line(self, wall_map, location, direction=0):
        lines = []
        if direction == 0: # Horizontal
            for w in range(self.house_min_w, self.house_max_w):
                if wall_map[location, w-1] == 0 and wall_map[location, w] == 127:
                    w_min = w
                if wall_map[location, w] == 127 and wall_map[location, w+1] == 0:
                    w_max = w
                    lines.append((w_min, w_max))
        else: # Vertical
            for h in range(self.house_min_h, self.house_max_h):
                if wall_map[h-1, location] == 0 and wall_map[h, location] == 127:
                    h_min = h
                if wall_map[h, location] == 127 and wall_map[h+1, location] == 0:
                    h_max = h
                    lines.append((h_min, h_max))   

        return lines

    def contact_length(self, line1, line2):
        length1 = line1[1] - line1[0] + 1
        length2 = line2[1] - line2[0] + 1
        length = max(line1[1], line2[1]) - min(line1[0], line2[0]) + 1
        contact_length = length1 + length2 - length
        return contact_length

    def find_contact_line(self, wall_map, location, input_line, direction=0):
        candidate_lines = self.scan_line(wall_map, location, direction=direction)
        contact_length = 0
        max_contact_line = (-1, -2)
        for line in candidate_lines:
            new_contact_length = self.contact_length(line, input_line)
            if new_contact_length > contact_length:
                max_contact_line = line
                contact_length = new_contact_length

        return max_contact_line 

    def find_contact_line2(self, wall_map, location, input_line, direction=0):
        contact_lines = []
        candidate_lines = self.scan_line(wall_map, location, direction=direction)
        for line in candidate_lines:
            if self.contact_length(line, input_line) > 0:
                contact_lines.append(line)
        return contact_lines 

    def wall_abstract(self, wall_map):
        walls = []
        while True:
            wall_block = []
            for h in range(self.house_min_h, self.house_max_h):
                lines = self.scan_line(wall_map, h, direction=0)
                if len(lines) > 0:
                    wall_block.append((h, lines[0]))
                    for w in range(lines[0][0], lines[0][1]+1):
                        wall_map[h, w] = 0
                    break

            if len(wall_block) == 0:
                break    

            i = 0
            while i < len(wall_block):
                h, input_line = wall_block[i][0], wall_block[i][1]
                contact_line_up = self.find_contact_line(wall_map, h-1, input_line)
                if contact_line_up[0] > 0:
                    wall_block.append((h-1, contact_line_up))
                    for w in range(contact_line_up[0], contact_line_up[1]+1):
                        wall_map[h-1, w] = 0
                contact_line_down = self.find_contact_line(wall_map, h+1, input_line)
                if contact_line_down[0] > 0:
                    wall_block.append((h+1, contact_line_down))
                    for w in range(contact_line_down[0], contact_line_down[1]+1):
                        wall_map[h+1, w] = 0
                i += 1

            merge_min_h = data_size
            merge_max_h = 0            
            merge_min_w = data_size
            merge_max_w = 0
            for block in wall_block:
                merge_min_h = block[0] if(merge_min_h > block[0]) else merge_min_h
                merge_max_h = block[0] if(merge_max_h < block[0]) else merge_max_h
                merge_min_w = block[1][0] if(merge_min_w > block[1][0]) else merge_min_w
                merge_max_w = block[1][1] if(merge_max_w < block[1][1]) else merge_max_w 
            merge_height = merge_max_h - merge_min_h + 1
            merge_width = merge_max_w - merge_min_w + 1  

            if merge_height != 1 and merge_width != 1 and merge_height*merge_width > 3*6:
                if merge_height > merge_width:                        
                    merge_mid_w = (merge_min_w + merge_max_w) // 2
                    merge_min_w =  merge_mid_w - 1
                    merge_max_w =  merge_mid_w + 1
                else:                        
                    merge_mid_h = (merge_min_h + merge_max_h) // 2
                    merge_min_h =  merge_mid_h - 1
                    merge_max_h =  merge_mid_h + 1

                merge_height = merge_max_h - merge_min_h + 1
                merge_width = merge_max_w - merge_min_w + 1                        
                if merge_height > merge_width:
                    walls.append([merge_min_h, merge_max_h, merge_min_w, merge_max_w, 1]) 
                else:    
                    walls.append([merge_min_h, merge_max_h, merge_min_w, merge_max_w, 0])

        return walls      

    def interior_wall_padding(self, tolerance=6):
        new_interior_wall_map = self.interior_wall_map.copy()
        for h in range(self.house_min_h, self.house_max_h):
            lines = self.scan_line(self.interior_wall_map, h, direction=0)
            if len(lines) > 1:
                for i in range(len(lines)-1):
                    line_0 = lines[i]
                    line_1 = lines[i+1]
                    if line_1[0] - line_0[1] - 1 < tolerance:
                        for w in range(line_0[1]+1, line_1[0]):
                            new_interior_wall_map[h, w] = 127

        for w in range(self.house_min_w, self.house_max_w):
            lines = self.scan_line(self.interior_wall_map, w, direction=1)
            if len(lines) > 1:
                for i in range(len(lines)-1):
                    line_0 = lines[i]
                    line_1 = lines[i+1]
                    if line_1[0] - line_0[1] - 1 < tolerance:
                        for h in range(line_0[1]+1, line_1[0]):
                            new_interior_wall_map[h, w] = 127

        for h in range(self.house_min_h, self.house_max_h):
            lines = self.scan_line(new_interior_wall_map, h, direction=0)
            for line in lines:
                contact_line_next = self.find_contact_line(new_interior_wall_map, h+1, line, direction=0)
                length_line = line[1] - line[0] + 1
                length_contact_line_next = contact_line_next[1] - contact_line_next[0] + 1
                if (length_contact_line_next > length_line > 0.5*length_contact_line_next) or \
                    (length_line > length_contact_line_next > 0.5*length_line):
                    min_w = min(line[0], contact_line_next[0])
                    max_w = max(line[1], contact_line_next[1])
                    for w in range(min_w, max_w+1):
                        new_interior_wall_map[h, w] = 127
                        new_interior_wall_map[h+1, w] = 127

        for w in range(self.house_min_w, self.house_max_w):
            lines = self.scan_line(new_interior_wall_map, w, direction=1)
            for line in lines:
                contact_line_next = self.find_contact_line(new_interior_wall_map, w+1, line, direction=1)
                length_line = line[1] - line[0] + 1
                length_contact_line_next = contact_line_next[1] - contact_line_next[0] + 1
                if (length_contact_line_next > length_line > 0.5*length_contact_line_next) or \
                    (length_line > length_contact_line_next > 0.5*length_line):
                    min_h = min(line[0], contact_line_next[0])
                    max_h = max(line[1], contact_line_next[1])
                    for h in range(min_h, max_h+1):
                        new_interior_wall_map[h, w] = 127
                        new_interior_wall_map[h, w+1] = 127
                        
        self.interior_wall_map = new_interior_wall_map

    def interior_wall_decomposition(self):
        # vertical decomposition is better than horizontal decomposition
        for w in range(self.house_min_w, self.house_max_w):
            lines = self.scan_line(self.interior_wall_map, w, direction=1)
            for line in lines:
                contact_line_nexts = self.find_contact_line2(self.interior_wall_map, w+1, line, direction=1)
                for contact_line_next in contact_line_nexts:
                    length_line = line[1] - line[0] + 1
                    length_contact_line_next = contact_line_next[1] - contact_line_next[0] + 1
                    if 0 < length_line < 0.5*length_contact_line_next:
                        for h in range(line[0], line[1]+1):
                            self.interior_wall_map[h, w] = 0
                    if 0 < length_contact_line_next < 0.5*length_line:
                        for h in range(line[0], line[1]+1):
                            self.interior_wall_map[h, w+1] = 0

    def merge(self, wall1, wall2):
        merge_min_h = min(wall1[0], wall2[0])
        merge_max_h = max(wall1[1], wall2[1])
        merge_min_w = min(wall1[2], wall2[2])
        merge_max_w = max(wall1[3], wall2[3])
        merge_height = merge_max_h - merge_min_h + 1
        merge_width = merge_max_w - merge_min_w + 1 
        if merge_height > merge_width:                        
            merge_mid_w = (merge_min_w + merge_max_w) // 2
            merge_min_w =  merge_mid_w - 1
            merge_max_w =  merge_mid_w + 1
            merge_wall = [merge_min_h, merge_max_h, merge_min_w, merge_max_w, 1]
        else:                        
            merge_mid_h = (merge_min_h + merge_max_h) // 2
            merge_min_h =  merge_mid_h - 1
            merge_max_h =  merge_mid_h + 1
            merge_wall = [merge_min_h, merge_max_h, merge_min_w, merge_max_w, 0]

        return merge_wall

    def interior_wall_merge(self, tolerance=5):
        i = 0
        while i < len(self.interior_walls):
            j = i + 1
            while j < len(self.interior_walls):
                flag = False           
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j] 
                height_wall1 = wall1[1] - wall1[0] + 1
                width_wall1 = wall1[3] - wall1[2] + 1
                height_wall2 = wall2[1] - wall2[0] + 1
                width_wall2 = wall2[3] - wall2[2] + 1  
                min_h = min(wall1[0], wall2[0])
                max_h = max(wall1[1], wall2[1])
                min_w = min(wall1[2], wall2[2])
                max_w = max(wall1[3], wall2[3])
                height = max_h - min_h + 1
                width = max_w - min_w + 1

                if wall1[4] == wall2[4] and height < height_wall1 + height_wall2 + tolerance and \
                    width < width_wall1 + width_wall2 + tolerance:
                    self.interior_walls[i] = self.merge(wall1, wall2)
                    self.interior_walls.pop(j)                    
                    flag = True                
                if flag is not True:
                    j = j + 1
            i = i + 1

    def interior_wall_adjustment(self):
        # wall-boundary alignment
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.exterior_boundary):
                wall = self.interior_walls[i]
                boundary = self.exterior_boundary[j]
                boundary_pre = self.exterior_boundary[j-1]
                if wall[0] - 6 < boundary[0] < wall[1] + 6 and wall[2] - 6 < boundary[1] < wall[3] + 6: 
                    if wall[4] == 0:
                        if (boundary_pre[2] == 0 and boundary[2] == 3) or (boundary_pre[2] == 1 and boundary[2] == 0):
                            self.interior_walls[i][0] = boundary[0] - 3
                            self.interior_walls[i][1] = boundary[0] - 1
                        if (boundary_pre[2] == 2 and boundary[2] == 1) or (boundary_pre[2] == 3 and boundary[2] == 2):
                            self.interior_walls[i][1] = boundary[0] + 2
                            self.interior_walls[i][0] = boundary[0]
                    if wall[4] == 1:
                        if (boundary_pre[2] == 0 and boundary[2] == 3) or (boundary_pre[2] == 3 and boundary[2] == 2):
                            self.interior_walls[i][2] = boundary[1] - 3
                            self.interior_walls[i][3] = boundary[1] - 1
                        if (boundary_pre[2] == 1 and boundary[2] == 0) or (boundary_pre[2] == 2 and boundary[2] == 1):
                            self.interior_walls[i][3] = boundary[1] + 2
                            self.interior_walls[i][2] = boundary[1]
                j = j + 1
            i = i + 1

        # wall alignment
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.interior_walls): 
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j] 
                height_wall1 = wall1[1] - wall1[0] + 1
                width_wall1 = wall1[3] - wall1[2] + 1
                height_wall2 = wall2[1] - wall2[0] + 1
                width_wall2 = wall2[3] - wall2[2] + 1  
                min_h = min(wall1[0], wall2[0])
                max_h = max(wall1[1], wall2[1])
                min_w = min(wall1[2], wall2[2])
                max_w = max(wall1[3], wall2[3])
                height = max_h - min_h + 1
                width = max_w - min_w + 1

                if wall1[4] == 0 and wall2[4] == 0:
                    if height <= height_wall1 + height_wall2 and width <= width_wall1 + width_wall2 + 9:
                        if width_wall1 < width_wall2:
                            self.interior_walls[i][0] = self.interior_walls[j][0]
                            self.interior_walls[i][1] = self.interior_walls[j][1]
                        else:
                            self.interior_walls[j][0] = self.interior_walls[i][0]
                            self.interior_walls[j][1] = self.interior_walls[i][1]

                if wall1[4] == 1 and wall2[4] == 1:
                    if height <= height_wall1 + height_wall2 + 9 and width <= width_wall1 + width_wall2:          
                        if height_wall1 < height_wall2:
                            self.interior_walls[i][2] = self.interior_walls[j][2]
                            self.interior_walls[i][3] = self.interior_walls[j][3]
                        else:
                            self.interior_walls[j][2] = self.interior_walls[i][2]
                            self.interior_walls[j][3] = self.interior_walls[i][3]
                j = j + 1
            i = i + 1

        # wall extension
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.interior_walls): 
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j]   

                if wall2[4] == 0 and wall1[4] == 1:
                    if wall1[2] >= wall2[2] and wall1[3] <= wall2[3]:
                        base_line = (wall2[0] + wall2[1]) // 2
                        distance_up = abs(wall1[0] - base_line)
                        distance_down = abs(wall1[1] - base_line)
                        if min(distance_up, distance_down) < 9 + 3:
                            if distance_up > distance_down:
                                self.interior_walls[i][1] = wall2[0]-1
                            else:
                                self.interior_walls[i][0] = wall2[1]+1
                    
                if wall1[4] == 0 and wall2[4] == 1:
                    if wall1[0] >= wall2[0] and wall1[1] <= wall2[1]:
                        base_line = (wall2[2] + wall2[3]) // 2
                        distance_left = abs(wall1[2] - base_line)
                        distance_right = abs(wall1[3] - base_line)
                        if min(distance_left, distance_right) < 9 + 3:
                            if distance_left > distance_right:
                                self.interior_walls[i][3] = wall2[2]-1
                            else:
                                self.interior_walls[i][2] = wall2[3]+1
                j = j + 1
            i = i + 1

        # corner alignment
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.interior_walls):  
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j]  

                if wall1[4] == 1 and wall2[4] == 0:                    
                    if wall1[2] == wall2[3]+1 or wall2[2] == wall1[3]+1:
                        if  wall2[0] - 6 < wall1[0] < wall2[1] + 6:
                            self.interior_walls[i][0] = wall2[0]
                        if  wall2[0] - 6 < wall1[1] < wall2[1] + 6:
                            self.interior_walls[i][1] = wall2[1]

                if wall1[4] == 0 and wall2[4] == 1:                    
                    if wall1[0] == wall2[1]+1 or wall2[0] == wall1[1]+1:
                        if  wall2[2] - 6 < wall1[2] < wall2[3] + 6:
                            self.interior_walls[i][2] = wall2[2]
                        if  wall2[2] - 6 < wall1[3] < wall2[3] + 6:
                            self.interior_walls[i][3] = wall2[3]
                j = j + 1
            i = i + 1 

        i = 0
        while i < len(self.interior_walls):
            j = 0 
            while j < len(self.interior_walls): 
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j]

                if wall1[4] == 1 and wall2[4] == 0:                    
                    if wall2[2] - 9 <= wall1[2] <= wall2[2]:
                        if wall2[0] <= wall1[0] <= wall2[1] + 9:
                            self.interior_walls[i][0] = wall2[1] + 1
                            self.interior_walls[j][2] = wall1[2]
                        if wall2[0] - 9 <= wall1[1] <= wall2[1]:
                            self.interior_walls[i][1] = wall2[0] - 1
                            self.interior_walls[j][2] = wall1[2]
                    if wall2[3] <= wall1[3] <= wall2[3] + 9:
                        if wall2[0] <= wall1[0] <= wall2[1] + 9:
                            self.interior_walls[i][0] = wall2[1] + 1
                            self.interior_walls[j][3] = wall1[3]
                        if wall2[0] - 9 <= wall1[1] <= wall2[1]:
                            self.interior_walls[i][1] = wall2[0] - 1
                            self.interior_walls[j][3] = wall1[3]

                if wall1[4] == 0 and wall2[4] == 1:                    
                    if wall2[0] - 9 <= wall1[0] <= wall2[0]:
                        if wall2[2] <= wall1[2] <= wall2[3] + 9:
                            self.interior_walls[i][2] = wall2[2]
                            self.interior_walls[j][0] = wall1[1] + 1
                        if wall2[2] - 9 <= wall1[3] <= wall2[3]:
                            self.interior_walls[i][3] = wall2[3]
                            self.interior_walls[j][0] = wall1[1] + 1
                    if wall2[1] <= wall1[1] <= wall2[1] + 9:
                        if wall2[2] <= wall1[2] <= wall2[3] + 9:
                            self.interior_walls[i][2] = wall2[2]
                            self.interior_walls[j][1] = wall1[0] - 1
                        if wall2[2] - 9 <= wall1[3] <= wall2[3]:
                            self.interior_walls[i][3] = wall2[3]
                            self.interior_walls[j][1] = wall1[0] - 1
                j = j + 1
            i = i + 1 

    def is_near_boundary(self, wall, dir, boundary, tolerance=15):

        if wall[4] == 0:
            if dir == 2:       
                count_sum = 0
                if self.interior_wall_map[wall[0] - 1][wall[2]] == 0 and self.inside_map[wall[0] - 1][wall[2]] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[0] + 1][wall[2] - 1] == 0 and self.inside_map[wall[0] + 1][wall[2] - 1] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[1] + 1][wall[2]] == 0 and self.inside_map[wall[1] + 1][wall[2]] == 255:
                    count_sum += 1
                if count_sum == 3 and 0 < wall[2] - boundary < tolerance:
                    return True
                else:
                    return False
            if dir == 3:
                count_sum = 0
                if self.interior_wall_map[wall[0] - 1][wall[3]] == 0 and self.inside_map[wall[0] - 1][wall[3]] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[0] + 1][wall[3] + 1] == 0 and self.inside_map[wall[0] + 1][wall[3] + 1] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[1] + 1][wall[3]] == 0 and self.inside_map[wall[1] + 1][wall[3]] == 255:
                    count_sum += 1
                if count_sum == 3 and 0 < boundary - wall[3] < tolerance:
                    return True
                else:
                    return False
        if wall[4] == 1:
            if dir == 0:
                count_sum = 0
                if self.interior_wall_map[wall[0]][wall[2] - 1] == 0 and self.inside_map[wall[0]][wall[2] - 1] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[0] - 1][wall[2] + 1] == 0 and self.inside_map[wall[0] - 1][wall[2] + 1] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[0]][wall[3] + 1] == 0 and self.inside_map[wall[0]][wall[3] + 1] == 255:
                    count_sum += 1
                if count_sum == 3 and 0 < wall[0] - boundary < tolerance:
                    return True
                else:
                    return False
            if dir == 1:
                count_sum = 0
                if self.interior_wall_map[wall[1]][wall[2] - 1] == 0 and self.inside_map[wall[1]][wall[2] - 1] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[1] + 1][wall[2] + 1] == 0 and self.inside_map[wall[1] + 1][wall[2] + 1] == 255:
                    count_sum += 1
                if self.interior_wall_map[wall[1]][wall[3] + 1] == 0 and self.inside_map[wall[1]][wall[3] + 1] == 255:
                    count_sum += 1
                if count_sum == 3 and 0 < boundary - wall[1] < tolerance:
                    return True
                else:
                    return False

    def is_break_wall(self, wall, dir):
        if wall[4] == 0:
            if dir == 2:
                flag = True
                for delta_h in range(-1, 2):
                    for delta_w in range(-1, 2):
                        if self.interior_wall_map[wall[0] - 2 + delta_h][wall[2] + 1 + delta_w] == 127 or self.inside_map[wall[0] - 2 + delta_h][wall[2] + 1 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[0] + 1 + delta_h][wall[2] - 2 + delta_w] == 127 or self.inside_map[wall[0] + 1 + delta_h][wall[2] - 2 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[1] + 2 + delta_h][wall[2] + 1 + delta_w] == 127 or self.inside_map[wall[1] + 2 + delta_h][wall[2] + 1 + delta_w] == 0:
                            flag = False
                return flag
            if dir == 3:
                flag = True
                for delta_h in range(-1, 2):
                    for delta_w in range(-1, 2):
                        if self.interior_wall_map[wall[0] - 2 + delta_h][wall[3] + 1 + delta_w] == 127 or self.inside_map[wall[0] - 2 + delta_h][wall[3] + 1 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[0] + 1 + delta_h][wall[3] + 2 + delta_w] == 127 or self.inside_map[wall[0] + 1 + delta_h][wall[3] + 2 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[1] + 2 + delta_h][wall[3] + 1 + delta_w] == 127 or self.inside_map[wall[1] + 2 + delta_h][wall[3] + 1 + delta_w] == 0:
                            flag = False
                return flag
        if wall[4] == 1:
            if dir == 0:
                flag = True
                for delta_h in range(-1, 2):
                    for delta_w in range(-1, 2):
                        if self.interior_wall_map[wall[0] + 1 + delta_h][wall[2] - 2 + delta_w] == 127 or self.inside_map[wall[0] + 1 + delta_h][wall[2] - 2 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[0] - 2 + delta_h][wall[2] + 1 + delta_w] == 127 or self.inside_map[wall[0] - 2 + delta_h][wall[2] + 1 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[0] + 1 + delta_h][wall[3] + 2 + delta_w] == 127 or self.inside_map[wall[0] + 1 + delta_h][wall[3] + 2 + delta_w] == 0:
                            flag = False
                return flag
            if dir == 1:
                flag = True
                for delta_h in range(-1, 2):
                    for delta_w in range(-1, 2):
                        if self.interior_wall_map[wall[1] - 1 + delta_h][wall[2] - 2 + delta_w] == 127 or self.inside_map[wall[1] - 1 + delta_h][wall[2] - 2 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[1] + 2 + delta_h][wall[2] + 1 + delta_w] == 127 or self.inside_map[wall[1] + 2 + delta_h][wall[2] + 1 + delta_w] == 0:
                            flag = False
                        if self.interior_wall_map[wall[1] - 1 + delta_h][wall[3] + 2 + delta_w] == 127 or self.inside_map[wall[1] - 1 + delta_h][wall[3] + 2 + delta_w] == 0:
                            flag = False
                return flag

    def interior_wall_final(self):
        interior_wall_map = np.zeros((data_size, data_size), dtype=np.uint8)
        for wall in self.interior_walls:
            for h in range(wall[0], wall[1]+1):
                for w in range(wall[2], wall[3]+1):
                    interior_wall_map[h, w] = 127
        self.interior_wall_map = interior_wall_map

        # wall-boundary extension
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.exterior_boundary):
                wall = self.interior_walls[i]
                boundary = self.exterior_boundary[j]
                boundary_pre = self.exterior_boundary[j-1]
                min_h = min(boundary[0], boundary_pre[0])
                max_h = max(boundary[0], boundary_pre[0])
                min_w = min(boundary[1], boundary_pre[1])
                max_w = max(boundary[1], boundary_pre[1])
                if wall[4] == 0 and min_w == max_w:
                    if wall[0] >= min_h and wall[1] <= max_h:
                        if self.is_near_boundary(wall, 2, min_w):
                            self.interior_walls[i][2] = min_w
                        if self.is_near_boundary(wall, 3, min_w):    
                            self.interior_walls[i][3] = min_w
                if wall[4] == 1 and min_h == max_h:
                    if wall[2] >= min_w and wall[3] <= max_w:
                        if self.is_near_boundary(wall, 0, min_h):
                            self.interior_walls[i][0] = min_h
                        if self.is_near_boundary(wall, 1, min_h):
                            self.interior_walls[i][1] = min_h
                j = j + 1
            i = i + 1

        # wall shrink
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.interior_walls): 
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j]   

                if wall2[4] == 0 and wall1[4] == 1:
                    if wall1[2] > wall2[2] and wall1[3] < wall2[3]:
                        base_line = (wall2[0] + wall2[1]) // 2
                        distance_up = abs(wall1[0] - base_line)
                        distance_down = abs(wall1[1] - base_line)
                        if min(distance_up, distance_down) < 9 + 3:
                            if distance_up > distance_down:
                                self.interior_walls[i][1] = wall2[0]-1
                            else:
                                self.interior_walls[i][0] = wall2[1]+1
                    
                if wall1[4] == 0 and wall2[4] == 1:
                    if wall1[0] > wall2[0] and wall1[1] < wall2[1]:
                        base_line = (wall2[2] + wall2[3]) // 2
                        distance_left = abs(wall1[2] - base_line)
                        distance_right = abs(wall1[3] - base_line)
                        if min(distance_left, distance_right) < 9 + 3:
                            if distance_left > distance_right:
                                self.interior_walls[i][3] = wall2[2]-1
                            else:
                                self.interior_walls[i][2] = wall2[3]+1
                j = j + 1
            i = i + 1

        # delete short wall
        i = 0
        while i < len(self.interior_walls):
            wall = self.interior_walls[i]
            flag = False
            if wall[4] == 0:
                if self.is_break_wall(wall, 2) and self.is_break_wall(wall, 3):
                    self.interior_walls.pop(i)
                    flag = True
                else:
                     if self.is_break_wall(wall, 2) or self.is_break_wall(wall, 3):
                        if wall[3] - wall[2] + 1 < 20:
                            self.interior_walls.pop(i)
                            flag = True
            else:
                if self.is_break_wall(wall, 0) and self.is_break_wall(wall, 1):
                    self.interior_walls.pop(i)
                    flag = True
                else:
                    if self.is_break_wall(wall, 0) or self.is_break_wall(wall, 1):
                        if wall[1] - wall[0] + 1 < 20:
                            self.interior_walls.pop(i)
                            flag = True
            if flag is not True:
                i = i + 1

        # wall shrink
        i = 0
        while i < len(self.interior_walls):
            j = 0
            while j < len(self.interior_walls): 
                wall1 = self.interior_walls[i]
                wall2 = self.interior_walls[j]   

                if wall2[4] == 0 and wall1[4] == 1:
                    if wall1[2] > wall2[2] and wall1[3] < wall2[3]:
                        base_line = (wall2[0] + wall2[1]) // 2
                        distance_up = abs(wall1[0] - base_line)
                        distance_down = abs(wall1[1] - base_line)
                        if min(distance_up, distance_down) < 9 + 3:
                            if distance_up > distance_down:
                                self.interior_walls[i][1] = wall2[0]-1
                            else:
                                self.interior_walls[i][0] = wall2[1]+1
                    
                if wall1[4] == 0 and wall2[4] == 1:
                    if wall1[0] > wall2[0] and wall1[1] < wall2[1]:
                        base_line = (wall2[2] + wall2[3]) // 2
                        distance_left = abs(wall1[2] - base_line)
                        distance_right = abs(wall1[3] - base_line)
                        if min(distance_left, distance_right) < 9 + 3:
                            if distance_left > distance_right:
                                self.interior_walls[i][3] = wall2[2]-1
                            else:
                                self.interior_walls[i][2] = wall2[3]+1
                j = j + 1
            i = i + 1

    def interior_wall_abstract(self):
        self.interior_wall_padding()
        self.interior_wall_decomposition()
        self.interior_walls = self.wall_abstract(self.interior_wall_map)
        self.interior_wall_merge(tolerance=5)
        self.interior_wall_adjustment()
        self.interior_wall_merge(tolerance=2)
        self.interior_wall_final()

    def exterior_boundary_abstract(self):
        # search direction:0(right)/1(down)/2(left)/3(up)
        flag = False
        for h in range(self.house_min_h, self.house_max_h):
            for w in range(self.house_min_w, self.house_max_w):
                if self.inside_map[h, w] == 255:
                    self.exterior_boundary.append((h, w, 0))
                    flag = True
                    break
            if flag:
                break

        while(flag):
            if self.exterior_boundary[-1][2] == 0:
                for w in range(self.exterior_boundary[-1][1]+1, self.house_max_w):
                    corner_sum = 0
                    if self.inside_map[self.exterior_boundary[-1][0], w] == 255:
                        corner_sum += 1
                    if self.inside_map[self.exterior_boundary[-1][0]-1, w] == 255:
                        corner_sum += 1
                    if self.inside_map[self.exterior_boundary[-1][0], w-1] == 255:
                        corner_sum += 1
                    if self.inside_map[self.exterior_boundary[-1][0]-1, w-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (self.exterior_boundary[-1][0], w, 1)
                        break
                    if corner_sum == 3:
                        new_point = (self.exterior_boundary[-1][0], w, 3)
                        break

            if self.exterior_boundary[-1][2] == 1:      
                for h in range(self.exterior_boundary[-1][0]+1, self.house_max_h): 
                    corner_sum = 0                
                    if self.inside_map[h, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside_map[h-1, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside_map[h, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if self.inside_map[h-1, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (h, self.exterior_boundary[-1][1], 2)
                        break
                    if corner_sum == 3:
                        new_point = (h, self.exterior_boundary[-1][1], 0)
                        break

            if self.exterior_boundary[-1][2] == 2:   
                for w in range(self.exterior_boundary[-1][1]-1, self.house_min_w, -1):
                    corner_sum = 0                     
                    if self.inside_map[self.exterior_boundary[-1][0], w] == 255:
                        corner_sum += 1
                    if self.inside_map[self.exterior_boundary[-1][0]-1, w] == 255:
                        corner_sum += 1
                    if self.inside_map[self.exterior_boundary[-1][0], w-1] == 255:
                        corner_sum += 1
                    if self.inside_map[self.exterior_boundary[-1][0]-1, w-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (self.exterior_boundary[-1][0], w, 3)
                        break
                    if corner_sum == 3:
                        new_point = (self.exterior_boundary[-1][0], w, 1)
                        break

            if self.exterior_boundary[-1][2] == 3:       
                for h in range(self.exterior_boundary[-1][0]-1, self.house_min_h, -1):
                    corner_sum = 0                
                    if self.inside_map[h, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside_map[h-1, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside_map[h, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if self.inside_map[h-1, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (h, self.exterior_boundary[-1][1], 0)
                        break
                    if corner_sum == 3:
                        new_point = (h, self.exterior_boundary[-1][1], 2)
                        break

            if new_point != self.exterior_boundary[0]:
                self.exterior_boundary.append(new_point)
            else:
                flag = False 
def MaskVector(floorplan):
    output_dir = 'synth_normalization'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    test_number = 0
    start_time = time.process_time()
    temp_time = start_time
    test_number = test_number + 1
    input_map = np.asarray(floorplan, dtype=np.uint8)
    output_map = np.zeros(input_map.shape, dtype=np.uint8)
    output_map[:,:,0] = input_map[:,:,0]
    output_map[:,:,2] = input_map[:,:,2]
    output_map[:,:,3] = input_map[:,:,3]
    abstracter = WallAbstract(input_map) 
    abstracter.exterior_boundary_abstract()
    abstracter.interior_wall_abstract() 
    for wall in abstracter.interior_walls:
        for h in range(wall[0], wall[1]+1):
            for w in range(wall[2], wall[3]+1):
                output_map[h, w, 1] = 127
    output = Image.fromarray(np.uint8(output_map))
    output.save(f'{output_dir}/result.png')            
    end_time = time.process_time()
    print(f'result: {(end_time-temp_time):.2f}s')
    temp_time = end_time
    end_time = time.process_time()
    cost_time = end_time-start_time
    print(f'Total test time: {cost_time:.2f}s')
    print(f'Total test number: {test_number}')
    print(f'Average time: {(cost_time/test_number):.2f}s')
    return output

