from gym_ras.tool.common import wrapAngleRange


print(wrapAngleRange(90+80, 10, 90))
print(wrapAngleRange(180, 10, 90))


print(wrapAngleRange(110, 10, 70))