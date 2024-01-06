import numpy as np
import torch

from scipy.spatial import ConvexHull

def sphericalFlip(points, center, param):
    n = points.shape[-2] # total n points
    points = points - np.repeat(center, n, axis = -2) # Move C to the origin
    normPoints = np.linalg.norm(points, axis = -1) # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
    R = np.repeat(np.max(normPoints, axis=-1) * np.power(10.0, param), n, axis = 0).reshape(-1, n) # Radius of Sphere
    flippedPointsTemp = 2*np.multiply(np.repeat((R - normPoints)[..., None], points.size()[-1], axis = -1), points)
    flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints[..., None], points.size()[-1], axis = -1)) # Apply Equation to get Flipped Points
    flippedPoints += points
    return flippedPoints
def hidden_point_removal(Xbd, encoding, cloud, campos):
    C = np.array([campos]).transpose(1, 0, -1) # View Point, which is well above the point cloud in z direction
    flippedCloud = sphericalFlip(cloud, C, np.pi) # Reflect the point cloud about a sphere centered at C
    zeros = np.zeros((flippedCloud.size()[0], 1, 3))
    points = np.concatenate((flippedCloud, zeros), axis=-2) # All points plus origin
    Xbd_empty = torch.full_like(Xbd, 256)
    encoding_empty = torch.full_like(encoding, 1024)
    vis_empty = torch.full_like(encoding, 512)
    for i in range(flippedCloud.size()[0]):
        hull = ConvexHull(points[i])  # Visibal points plus possible origin. Use its vertices property.
        vis = hull.vertices[:-1]  # remove origin
        Xbd_empty[i][vis] = Xbd[i][vis]
        encoding_empty[i][vis] = encoding[i][vis]
        vis_empty[i][:len(vis)] = torch.from_numpy(vis)
    return Xbd_empty.cuda(), encoding_empty.cuda(), vis_empty.cuda()

def sample_sphere(point_N):
    views = np.array([[1., 0, 0], [-1, 0, 0], [0, 1., 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

    random_views = views[np.random.choice(len(views), point_N, replace=True)]
    return random_views