from air_violin.types import Hand, Point, TrackingFrame


def hand(label="Left", wrist=(0.7, 0.5), finger=(0.28, 0.43), pinch=1.0, confidence=0.95):
    points = [Point(0.5, 0.5) for _ in range(21)]
    points[0] = Point(*wrist)
    points[5], points[17] = Point(0.5, 0.5), Point(0.6, 0.5)
    points[8] = Point(*finger)
    points[4] = Point(points[8].x + pinch * 0.1, points[8].y)
    return Hand(label, confidence, tuple(points))


def frame(t, wrist_x=0.7, finger=(0.28, 0.43), pinch=1.0, left=True, right=True):
    hands = []
    if left:
        hands.append(hand("Left", finger=finger))
    if right:
        hands.append(hand("Right", wrist=(wrist_x, 0.5), pinch=pinch))
    return TrackingFrame(t, tuple(hands))
