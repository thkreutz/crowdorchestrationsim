import pygame
import math

### Definition of some obstacles. Polyline objects and Circle-like obstacles

# Obstacles
class Obstacle():
    int_to_class = { 0: "Undefined", 1 : "wall", 2 : "railing", 3 : "polyline", 4 : "obstacle", 10 : "pedestrian"}
    class_to_int = { "Undefined" : 0 , "wall" : 1 , "railing" : 2, "polyline" : 3, "obstacle" : 4, "pedestrian" : 10}
    uid = 1
    def __init__(self, x, y, semantic_class=0):
        self.x = x
        self.y = y
        self.semantic_class = semantic_class
        self.id = Obstacle.uid
        Obstacle.uid += 1

    def get_semantic_class(self):
        return self.int_to_class[self.semantic_class]


class Circle(Obstacle):
    def __init__(self, x, y, radius, semantic_class=0):
        Obstacle.__init__(self, x, y, semantic_class)
        self.radius = radius
        # calculate the bounding box of the circle


    def draw(self, pygame, window):
        pygame.draw.circle(window, (0, 200, 200), (self.x, self.y), self.radius, 5)

    def check_collision(self, obstacle):
        # if instnace of Polyline -> Rect collision
        if isinstance(obstacle, Polyline):
            #return self.collides_with_rect(obstacle.rect)
            return self.circle_polygon_collision((self.x, self.y), self.radius, obstacle.points)
            #return self.collides_with_polygon(obstacle)
        
        # if instance of Circle -> Circle collision
        if isinstance(obstacle, Circle):
            return self.collides_with_circle(obstacle)

        return False

    def collides_with_circle(self, other_circle):
        distance = ((self.x - other_circle.x) ** 2 + (self.y - other_circle.y) ** 2) ** 0.5
        return distance < (self.radius + other_circle.radius)
    

    # Define a function to check if a circle is colliding with a polygon
    def circle_polygon_collision(self, circle_pos, circle_radius, polygon_vertices):
        # Check for collision with the vertices of the polygon
        for vertex in polygon_vertices:
            distance = math.sqrt((vertex[0] - circle_pos[0]) ** 2 + (vertex[1] - circle_pos[1]) ** 2)
            if distance < circle_radius:
                return True

        # Check for collision with the edges of the polygon
        for i in range(len(polygon_vertices)):
            j = (i + 1) % len(polygon_vertices)
            edge = [polygon_vertices[j][0] - polygon_vertices[i][0], polygon_vertices[j][1] - polygon_vertices[i][1]]
            edge_length = math.sqrt(edge[0] ** 2 + edge[1] ** 2)
            u = [edge[0] / edge_length, edge[1] / edge_length]
            circle_center_to_vertex1 = [circle_pos[0] - polygon_vertices[i][0], circle_pos[1] - polygon_vertices[i][1]]
            projection_length = u[0] * circle_center_to_vertex1[0] + u[1] * circle_center_to_vertex1[1]
            if projection_length < 0 or projection_length > edge_length:
                continue
            closest_point_on_edge = [polygon_vertices[i][0] + projection_length * u[0], polygon_vertices[i][1] + projection_length * u[1]]
            distance = math.sqrt((circle_pos[0] - closest_point_on_edge[0]) ** 2 + (circle_pos[1] - closest_point_on_edge[1]) ** 2)
            if distance < circle_radius:
                return True

        return False

    def collides_with_polygon(self, polygon):
        
        # Separating Axis Theorem (SAT)... 
        
        collides = False
        if self.collides_with_rect(polygon.rect):
        #    for vertex in polygon.points:
        #        if math.sqrt((vertex[0] - self.x)**2 + (vertex[1] - self.y)**2) <= self.radius:
        #            collides = True
        
            # Check if any of the polygon edges intersect with the circle
            for i in range(len(polygon.points)):
                p1 = tuple(polygon.points[i])
                p2 = tuple(polygon.points[(i+1)%len(polygon.points)])
                print(p1, p2)
                edge = pygame.math.Vector2(p2) - pygame.math.Vector2(p1)
                axis = edge.normalize()

                # Project the polygon and the circle onto the axis
                polygon_points = [pygame.math.Vector2(tuple(v) ) for v in polygon.points]
                projected_polygon_points = [p.dot(axis) for p in polygon_points]
                projected_circle_pos = pygame.math.Vector2((self.x, self.y)).dot(axis)
                min_polygon_point = min(projected_polygon_points)
                max_polygon_point = max(projected_polygon_points)
                min_circle_point = projected_circle_pos - self.radius
                max_circle_point = projected_circle_pos + self.radius

                # Check for overlap
                if max_polygon_point < min_circle_point or max_circle_point < min_polygon_point:
                    break  # No overlap, exit early

            else:
                # No separating axis found, collision detected
                collides = True

        return collides  #or any([pygame.math.Vector2(v) for v in polygon.vertices]).distance_to(pygame.math.Vector2((self.x, self.y))) < self.radius

    def collides_with_rect(self, rect):
        rect_center = (rect.left + rect.width / 2, rect.top + rect.height / 2)
        dx = abs(self.x - rect_center[0])
        dy = abs(self.y - rect_center[1])
        if dx > rect.width / 2 + self.radius:
            return False
        if dy > rect.height / 2 + self.radius:
            return False
        if dx <= rect.width / 2:
            return True
        if dy <= rect.height / 2:
            return True
        corner_distance = (dx - rect.width / 2) ** 2 + (dy - rect.height / 2) ** 2
        return corner_distance <= (self.radius ** 2)


class Polyline(Obstacle):
    def __init__(self, points, semantic_class=0):
        x, y = points[0]
        Obstacle.__init__(self, x, y, semantic_class)
        self.points = points
        self.drawn = False
        self.rect = self.make_rect_from_polyline(self.points)

    def draw(self, pygame, window):
        #pygame.draw.lines(window, (0, 100, 0), closed=True, points=self.points, width=10)
        self.polygon = pygame.draw.polygon(window, (0, 100, 0), points=self.points, width=3)
        self.drawn = True

    def make_rect_from_polyline(self, polyline):
        # Find the minimum and maximum x and y values of the points in the polyline
        min_x = min(point[0] for point in polyline)
        max_x = max(point[0] for point in polyline)
        min_y = min(point[1] for point in polyline)
        max_y = max(point[1] for point in polyline)

        # Create a Rect object that encompasses all the points in the polyline
        rect_pos = (min_x, min_y)
        rect_size = (max_x - min_x, max_y - min_y)
        return pygame.Rect(rect_pos, rect_size)
