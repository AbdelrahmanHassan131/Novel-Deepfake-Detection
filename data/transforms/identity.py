"""Module-level no-op transform, serializable by spawned DataLoader workers."""


def identity_image(image):
    return image
