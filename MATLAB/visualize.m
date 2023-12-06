clear;
clc;

close all;

pc_file = fopen('000772.bin', 'rb');
raw_data = fread(pc_file, 'float32');
fclose(pc_file);

x = raw_data(1:4:end);
y = raw_data(2:4:end);
z = raw_data(3:4:end);

pc = pointCloud([x y z]);

pcshow(pc);
