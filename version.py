VERSION = (0, 4, 2)
COMPATIBILITY_BUILD = "0.4.2-compat.1"
UPSTREAM_VERSION = "0.4.1"
UPSTREAM_COMMIT = "c2622a8a9f1ae6b790cfe1d2571f814b126811b4"
def version_tag(version):
    return f"{version[0]}.{version[1]}.{version[2]}"

def version_tuple(tag):
    return tuple(map(lambda x: int(x), tag.split('.')))
