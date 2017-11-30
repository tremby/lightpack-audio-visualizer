from __future__ import division
import numpy
import audioop
import matplotlib.pyplot as plot
import math
import sys
import imp
import os
import random
from colour import Colour
import lightpack
import copy

# TODO: store the true max and min seen as limits, but when normalizing only use
# the interquartile range or so
# TODO: test square waves at 1% and 100% amplitude
# TODO: test sample with period of total silence

LIGHTPACK_API_KEY = 'hunter2'

INFILE = '/tmp/mpd.fifo'

FREQUENCY_BUCKETS = 5

HEARING_RANGE = (30, 18e3)

AUDIO_RATE = 44.1e3
AUDIO_DEPTH = 16 # This can't be changed without changing the numpy.frombuffer calls
AUDIO_CHANNELS = 2

BITS_IN_BYTE = 8

BUFFER_SAMPLES = 4096
REFRESH_BUFFER_FACTOR = 16

# Only push the limits if the measured value is greater than the existing limit
# by a certain factor, but not greater than the existing limit by another factor
LIMIT_PUSH_THRESHOLDS = ((-0.05, -0.3), (1.02, 1.2))

BUFFER_SIZE = BUFFER_SAMPLES * AUDIO_CHANNELS * AUDIO_DEPTH // BITS_IN_BYTE

READ_FREQUENCY = REFRESH_BUFFER_FACTOR * AUDIO_RATE / BUFFER_SAMPLES

def get_audio():
	"""
	Get a chunk of audio.

	The first time this reads the full number of samples required for the FFT.
	Subsequent calls give a major latter chunk of the previous output plus the
	new samples required to fill the rest of the buffer.
	"""
	buf = None
	num_new_bytes = BUFFER_SIZE // REFRESH_BUFFER_FACTOR
	with open(INFILE) as fifo:
		while True:
			if buf is None:
				buf = fifo.read(BUFFER_SIZE)
			else:
				buf = buf[num_new_bytes:] + fifo.read(num_new_bytes)
			yield buf

def frequency_bucket_floor(bucket_index):
	"""
	Get lowest frequency to be put in a particular frequency bucket
	"""
	fraction = bucket_index / FREQUENCY_BUCKETS
	log_range = [math.log(edge, 2) for edge in HEARING_RANGE]
	log_floor = log_range[0] + fraction * (log_range[1] - log_range[0])
	return 2 ** log_floor

def fft_index(fft, frequency):
	"""
	Index in fft for a particular frequency in Hz
	"""
	return 2 * int(len(fft) * frequency / AUDIO_RATE) # Not entirely clear on why I need to multiply by 2 here. I don't need to if I use fft instead of rfft, but then I get a bunch of crazy high frequency FFT data, or is it complex numbers or something...

def fft_frequency(fft, index):
	"""
	Frequency for a given index in the fft
	"""
	return index * AUDIO_RATE / len(fft) / 2 # Same as in fft_index, see above

def split_into_channels(audio):
	"""
	Separate a buffer of audio into its separate channels
	"""
	if AUDIO_CHANNELS == 1:
		return [audio]
	if AUDIO_CHANNELS == 2:
		return [audioop.tomono(audio, AUDIO_DEPTH // BITS_IN_BYTE, 1, 0),
				audioop.tomono(audio, AUDIO_DEPTH // BITS_IN_BYTE, 0, 1)]
	raise NotImplementedError("Only 1 or 2 channels are supported")

def parse_to_numpy(audio):
	"""
	Take a buffer of audio and convert it into a numpy structure
	"""
	return numpy.frombuffer(audio, dtype=numpy.int16)

def apply_window(audio):
	"""
	Take a numpy structure and apply a Hanning window to it
	"""
	return audio * numpy.hanning(len(audio))

def apply_fft(audio):
	"""
	Get absolute fast Fourier transform of audio data
	"""
	return numpy.abs(numpy.fft.rfft(audio))

audio_getter = get_audio()

hearing_span = HEARING_RANGE[1] - HEARING_RANGE[0]

# Get bucket frequency spans
bucket_spans = [(frequency_bucket_floor(bucket_index), \
			frequency_bucket_floor(bucket_index + 1)) \
		for bucket_index in range(FREQUENCY_BUCKETS)]

# meter_limits = [[float('inf'), -float('inf')] for i in range(FREQUENCY_BUCKETS)]
BASE_METER_LIMITS = [[2.0, 6.7], [2.5, 6.5], [2.3, 5.7], [2.1, 5.4], [1.4, 4.8]] # A little above (lower limit) and below (upper limit) observed limits
meter_limits = copy.deepcopy(BASE_METER_LIMITS)

# observed limits:
# 0: [1.9576902952073380, 6.8014675519811183]
# 1: [2.4070367367522851, 6.5660445868585837]
# 2: [2.2746862664216541, 5.7502221752346649]
# 3: [2.0502071988334905, 5.4596701447828329]
# 4: [1.3541526014206986, 4.8294415858312494]

lp = lightpack.Lightpack(api_key=LIGHTPACK_API_KEY, led_map=range(10))
lp.connect()
lp.lock()
lp.setColourToAll((0, 0, 0))

def exit(code=0):
	"""
	Disconnect Lightpack and exit
	"""
	try:
		lp.disconnect()
	except:
		pass
	sys.exit(code)

lightmap = [
	[9, 8, 7, 6, 5], # Left channel low to high frequency
	[0, 1, 2, 3, 4], # Right channel low to high frequency
]

def randomize_spectrum_angle():
	val = random.random() * 360
	print "spectrum angle is %s " % val
	return val

def randomize_shift_period():
	val = random.triangular(1, 20, 4)
	val = val * (1 if random.random() > 0.5 else -1)
	print "shift period is %s" % val
	return val

hue = 0
shift_period = randomize_shift_period()
spectrum_angle = randomize_spectrum_angle()
exponent = 6
randomized = True

iteration = 0

try:
	while True:
		iteration += 1

		# Get next buffer of audio to work with
		audio = audio_getter.next()

		if len(audio) is 0:
			# End of audio data
			print "original limits: %s" % BASE_METER_LIMITS
			print "final limits: %s" % meter_limits
			break

		# Split into two channels
		audio = split_into_channels(audio)

		# Turn into numpy structures
		audio = map(parse_to_numpy, audio)

		# Apply window
		audio = map(apply_window, audio)

		# Calculate real fast fourier transform
		fft = map(apply_fft, audio)

		normalized_values = []

		# For each channel
		try:
			# print
			# For each frequency bucket
			lights = []
			for bucket_index in range(FREQUENCY_BUCKETS):
				bucket_span = bucket_spans[bucket_index]

				for channel_index in range(AUDIO_CHANNELS):
					index_span = [fft_index(fft[channel_index], i) for i in bucket_span]

					# Get samples we care about
					bucket = fft[channel_index][index_span[0]:index_span[1]]

					# Get RMS of this bucket
					bucket_rms = numpy.sqrt(numpy.mean(bucket ** 2))

					if bucket_rms == 0:
						continue

					# Put value on a log scale
					value = numpy.log10(bucket_rms)

					if value < 0:
						print "value %s below zero; skipping. what does this mean?" % value
						continue

					# Apply lower and upper limits to get on a scale nominally
					# from 0 to 1
					normalized_value = value - meter_limits[bucket_index][0]
					normalized_value /= meter_limits[bucket_index][1] - meter_limits[bucket_index][0]

					# Update the limits if applicable
					if normalized_value < 0:
						# print "new low value %s (normalized %s), limit was %s" % (value, normalized_value, meter_limits[bucket_index][0])
						# print "value %s, limits %s" % (value, meter_limits[bucket_index])
						if normalized_value > LIMIT_PUSH_THRESHOLDS[0][0]:
							# print "ignore, since it is before the push range %s" % LIMIT_PUSH_THRESHOLDS[0][0]
							pass
						elif normalized_value < LIMIT_PUSH_THRESHOLDS[0][1]:
							# print "ignore, since it is beyond the push range %s" % LIMIT_PUSH_THRESHOLDS[0][1]
							pass
						else:
							# print "it is in the push range"
							meter_limits[bucket_index][0] = value
					# meter_limits[bucket_index][0] = min(meter_limits[bucket_index][0], normalized_value)
					elif normalized_value > 1:
						# print "new high value %s (normalized %s), limit was %s" % (value, normalized_value, meter_limits[bucket_index][1])
						if normalized_value < LIMIT_PUSH_THRESHOLDS[1][0]:
							# print "ignore, since it is before the push range"
							pass
						elif normalized_value > LIMIT_PUSH_THRESHOLDS[1][1]:
							# print "ignore, since it is beyond the push range"
							pass
						else:
							# print "it is in the push range"
							meter_limits[bucket_index][1] = value
					# meter_limits[bucket_index][1] = max(meter_limits[bucket_index][1], normalized_value)

					# Clip the value to be between 0 and 1
					normalized_value = max(0, min(1, normalized_value))

					if math.isnan(normalized_value):
						print "value is not a number; skipping. what does this mean?"
						continue

					normalized_values.append(normalized_value)

					if not iteration % 200:
						# print "%s: %s, %s" % (bucket_index, meter_limits[bucket_index], normalized_value)
						pass

					colour = Colour().hsl((hue + spectrum_angle * bucket_index / FREQUENCY_BUCKETS, 1, normalized_value ** exponent))
					lights.append((lightmap[channel_index][bucket_index], colour))
			lp.setColours(*lights)
		except OverflowError:
			continue

		if numpy.mean(normalized_values) < 0.05:
			if not randomized:
				randomized = True
				print "gap in music"
				hue = 360 * random.random()
				shift_period = randomize_shift_period()
				spectrum_angle = randomize_spectrum_angle()
		else:
			randomized = False
			hue += 360 / READ_FREQUENCY / shift_period
			hue = hue % 360

except KeyboardInterrupt:
	exit(0)
