import removebackground
import vectorization
import synth
data = {
    "mask": [[140, 126], [140, 224], [249, 224], [249, 168], [247, 168], [247, 70], [217, 70], [217, 140], [175, 140], [175, 126]],
    "door_pos": [233, 70],
    "area": 117.5
}

mask=removebackground.CreateMask(data)
img=synth.LiSynth(mask)
vectorization.MaskVector(img)