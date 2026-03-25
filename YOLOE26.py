#Ultralytics YOLOE-26 - New Promptable segmentation model
# Improved version of original YOLOE model.

from ultralytics import YOLOE


# Prompt free (YOLOE-26 model)
model = YOLOE("yoloe-26l-seg-pf.pt")
results = model.predict("video.mp4",
                        show=True,
                        save=True,
                        show_conf=False)
# for r in results:  # Extract the results
#     boxes = r.boxes.xyxy.cpu().tolist()
#     cls = r.boxes.cls.cpu().tolist()
#     for b, c in zip(boxes, cls): # display results
#         print(f"box: {b}, cls: {model.names[c]}")


# Text prompt example #01 (YOLOE-26 model)
# model = YOLOE("yoloe-26x-seg.pt")
# names = ["white tiger"]  # you can also try name="animal" or "tiger walking"
# model.set_classes(names, model.get_text_pe(names))
# model.predict("tiger.mp4",
#               show=True,
#               save=True)


# Text prompt example #02 (YOLOE-26 model)
# model = YOLOE("yoloe-26x-seg.pt")
# names = ["white horse"]  # you can also try name="animal" or "black horse"
# model.set_classes(names, model.get_text_pe(names))
# model.predict("horse.mp4",
#               show=True,
#               save=True)


# Text prompt example #03 (YOLOE-26 model)
# model = YOLOE("yoloe-26x-seg.pt")
# names = ["horse in water"]  # you can also try name="black horse", "black hat"
# model.set_classes(names, model.get_text_pe(names))
# model.predict("person-on-horse.mp4",
#               show=True,
#               save=True)


# Text prompt example #04 (YOLOE-26 model)
# model = YOLOE("yoloe-26x-seg.pt")
# names = ["falcon", "eagle", "pigeon", "blue jay"]  # you can also try name="falcon flying"
# model.set_classes(names, model.get_text_pe(names))
# model.predict("birds.mp4",
#               show=True,
#               save=True)