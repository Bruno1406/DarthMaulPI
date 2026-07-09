class Cube:
    def __init__(self, x_cm, y_cm, color_id):
        self.x_cm = round(x_cm)
        self.y_cm = round(y_cm)
        self.seen_count = 1
        self.color_votes = {}
        self.add_color_vote(color_id)

    def add_color_vote(self, color_id):
        if color_id not in self.color_votes:
            self.color_votes[color_id] = 0
        self.color_votes[color_id] += 1

    def distance_to(self, x_cm, y_cm):
        dx = self.x_cm - x_cm
        dy = self.y_cm - y_cm
        return (dx * dx + dy * dy) ** 0.5

    def update(self, x_cm, y_cm, color_id):
        self.seen_count += 1
        self.x_cm = round((self.x_cm * (self.seen_count - 1) + x_cm) / self.seen_count)
        self.y_cm = round((self.y_cm * (self.seen_count - 1) + y_cm) / self.seen_count)
        self.add_color_vote(color_id)

    def get_color_id(self):
        return max(self.color_votes, key=self.color_votes.get)



class CubeTracker:
    def __init__(self, limit_distance_cm=20):
        self.cubes = []
        self.limit_distance_cm = limit_distance_cm

    def add_cube(self, x_cm, y_cm, color_id):
        for cube in self.cubes:
            if cube.distance_to(x_cm, y_cm) <= self.limit_distance_cm:
                cube.update(x_cm, y_cm, color_id)
                return False

        cube = Cube(x_cm, y_cm, color_id)
        self.cubes.append(cube)
        return True

    def export_for_service(self):
        n = len(self.cubes)
        xs = []
        ys = []
        colors = []

        for cube in self.cubes:
            xs.append(cube.x_cm)
            ys.append(cube.y_cm)
            colors.append(cube.get_color_id())

        return n,xs,ys,colors
    
    def get_cube_count(self):
        return len(self.cubes)