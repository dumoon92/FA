function my_plot(x, y, ysvm, ykrig, N, kernel)
figure
hold on
plot(x,y,'x','LineWidth',2)
plot(x,[ysvm,ykrig],'LineWidth',2)
hold off
grid on
axis square
xlabel('x');
ylabel('y');
legend('humps(x)','SVM fit','Kriging fit');
title(['Simulation for ', num2str(N), ' Data points using ' kernel, ' as SVM Kernel'])
