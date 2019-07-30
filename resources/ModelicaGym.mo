package ModelicaGym
    model CartPole "Classic cart pole simulated with an inverted pendulum"
                parameter Modelica.SIunits.Mass m_cart = 10;
                parameter Modelica.SIunits.Mass m_pole = 1;
                parameter Modelica.SIunits.Length length = 2;
                parameter Modelica.SIunits.Angle theta_0 = 85/180*pi;
                parameter Modelica.SIunits.AngularVelocity theta_dot_0 = 0.0;
                constant Real pi = Modelica.Constants.pi;
                inner Modelica.Mechanics.MultiBody.World world(gravityType=Modelica.Mechanics.MultiBody.Types.GravityTypes.
                            UniformGravity, animateWorld=false)
                                                            annotation (Placement(visible = true, transformation(origin = {-130, -70}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
                Modelica.Mechanics.MultiBody.Joints.Prismatic prismatic(useAxisFlange=true)
                    annotation (Placement(transformation(extent={{-96,0},{-76,20}})));
                Modelica.Mechanics.Translational.Components.Damper damper1(d=0)
                    annotation (Placement(transformation(extent={{-96,14},{-76,34}})));
                Modelica.Mechanics.MultiBody.Joints.Revolute rev(n={0,0,1},useAxisFlange=true,
                    phi(fixed=true, start=theta_0),
                    w(fixed=true, start=theta_dot_0))
                                                                          annotation (Placement(transformation(extent={{-30,0},
                                    {-10,20}},      rotation=0)));
                Modelica.Mechanics.Rotational.Components.Damper damper(d=0)
                    annotation (Placement(transformation(extent={{-22,40},{-2,60}},rotation=0)));
                Modelica.Mechanics.MultiBody.Parts.Body poleMassCenter(
                    m=m_pole,
                    r_CM={0,0,0},
                    specularCoefficient=4*world.defaultSpecularCoefficient,
                    sphereDiameter=1.5*world.defaultBodyDiameter)
                    annotation (Placement(transformation(extent={{78,0},{98,20}}, rotation=0)));
                Modelica.Mechanics.MultiBody.Parts.BodyShape cart(
                    shapeType="box",
                    animateSphere=true,
                    m=m_cart,
                    sphereDiameter=world.defaultBodyDiameter,
                    r={0,0,0},
                    r_CM={0,0,0})
                    annotation (Placement(transformation(extent={{-58,0},{-38,20}})));
                Modelica.Mechanics.Translational.Sources.Force force
                    annotation (Placement(transformation(extent={{-98,34},{-78,54}})));
                Modelica.Mechanics.MultiBody.Sensors.RelativeAngles relativeAngles
                    annotation (Placement(transformation(extent={{-30,-30},{-10,-10}})));
                Modelica.Mechanics.MultiBody.Sensors.RelativeVelocity relativeVelocity
                    annotation (Placement(transformation(extent={{-96,-30},{-76,-10}})));
                Modelica.Mechanics.MultiBody.Sensors.RelativePosition relativePosition
                    annotation (Placement(visible = true, transformation(extent = {{-96, -60}, {-76, -40}}, rotation = 0)));
                Modelica.Blocks.Interfaces.RealInput f
                    annotation (Placement(transformation(extent={{-190,-20},{-150,20}})));
                Modelica.Blocks.Interfaces.RealOutput x
                    annotation (Placement(visible = true, transformation(extent = {{150, 70}, {170, 90}}, rotation = 0), iconTransformation(extent = {{150, 70}, {170, 90}}, rotation = 0)));
                Modelica.Blocks.Interfaces.RealOutput x_dot
                    annotation (Placement(transformation(extent={{150,50},{170,70}})));
                Modelica.Blocks.Interfaces.RealOutput theta_dot
                    annotation (Placement(transformation(extent={{150,-60},{170,-40}})));
                Modelica.Mechanics.MultiBody.Sensors.RelativeAngularVelocity
                    relativeAngularVelocity
                    annotation (Placement(transformation(extent={{-30,-60},{-10,-40}})));
                Modelica.Mechanics.MultiBody.Parts.BodyCylinder poleCartConnection(
                    r={length/2,0,0},
                    specularCoefficient=0.7,
                    color={0,0,0},
                    diameter=0.05,
                    density=900)
                    "half of the pole length" annotation(
      Placement(transformation(extent = {{26, 0}, {46, 20}})));
              Modelica.Blocks.Interfaces.RealOutput theta
                    annotation (Placement(transformation(extent={{150,6},{170,26}})));
            equation
        connect(relativePosition.r_rel[1], x) annotation(
            Line(points = {{-86, -61}, {-104, -61}, {-104, -58}, {-120, -58}, {-120, 80}, {160, 80}}, color = {0, 0, 127}));
        connect(relativePosition.frame_a, relativeVelocity.frame_a) annotation(
            Line(points = {{-96, -50}, {-96, -20}}, color = {95, 95, 95}, thickness = 0.5));
        connect(relativePosition.frame_b, relativeVelocity.frame_b) annotation(
            Line(points = {{-76, -50}, {-76, -20}}, color = {95, 95, 95}, thickness = 0.5));
                connect(prismatic.frame_a, world.frame_b) annotation (Line(
                        points={{-96,10},{-110,10}, {-110, -70}, {-120, -70}},
                        color={95,95,95},
                        thickness=0.5));
                connect(damper.flange_b, rev.axis) annotation (Line(points={{-2,50},{0,50},{0,
                                24},{0,20},{-20,20}},   color={0,0,0}));
                connect(rev.support, damper.flange_a) annotation (Line(points={{-26,20},{-26,
                                26},{-36,26},{-36,50},{-22,50}}, color={0,0,0}));
  connect(cart.frame_b, rev.frame_a) annotation(
      Line(points = {{-38, 10}, {-30, 10}}, color = {95, 95, 95}, thickness = 0.5, smooth = Smooth.None));
                connect(force.flange, prismatic.axis) annotation (Line(
                        points={{-78,44},{-78,16}},
                        color={0,127,0},
                        smooth=Smooth.None));
                connect(damper1.flange_a, prismatic.support) annotation (Line(
                        points={{-96,24},{-96,16},{-90,16}},
                        color={0,127,0},
                        smooth=Smooth.None));
                connect(damper1.flange_b, prismatic.axis) annotation (Line(
                        points={{-76,24},{-78,24},{-78,16}},
                        color={0,127,0},
                        smooth=Smooth.None));
  connect(prismatic.frame_b, cart.frame_a) annotation(
      Line(points = {{-76, 10}, {-58, 10}}, color = {95, 95, 95}, thickness = 0.5, smooth = Smooth.None));
                connect(relativeVelocity.frame_b, prismatic.frame_b) annotation (Line(
                        points={{-76,-20},{-76,10}},
                        color={95,95,95},
                        thickness=0.5,
                        smooth=Smooth.None));
                connect(relativeVelocity.frame_a, prismatic.frame_a) annotation (Line(
                        points={{-96,-20},{-96,10}},
                        color={95,95,95},
                        thickness=0.5,
                        smooth=Smooth.None));
                connect(relativeAngles.frame_b, rev.frame_b) annotation (Line(
                        points={{-10,-20},{-10,10}},
                        color={95,95,95},
                        thickness=0.5,
                        smooth=Smooth.None));
                connect(relativeAngles.frame_a, rev.frame_a) annotation (Line(
                        points={{-30,-20},{-30,10}},
                        color={95,95,95},
                        thickness=0.5,
                        smooth=Smooth.None));
                connect(f, force.f) annotation (Line(
                        points={{-170,0},{-136,0},{-136,44},{-100,44}},
                        color={0,0,127},
                        smooth=Smooth.None));
                connect(relativeAngularVelocity.frame_a, relativeAngles.frame_a) annotation (
                        Line(
                        points={{-30,-50},{-30,-20}},
                        color={95,95,95},
                        thickness=0.5,
                        smooth=Smooth.None));
                connect(relativeAngularVelocity.frame_b, relativeAngles.frame_b) annotation (
                        Line(
                        points={{-10,-50},{-10,-20}},
                        color={95,95,95},
                        thickness=0.5,
                        smooth=Smooth.None));
                connect(relativeAngularVelocity.w_rel[3], theta_dot) annotation (Line(
                        points={{-20,-61.6667},{-20,-58},{100,-58},{100,-50},{160,-50}},
                        color={0,0,127},
                        smooth=Smooth.None));
                connect(relativeVelocity.v_rel[1], x_dot) annotation (Line(
                        points={{-86,-30.3333},{-100,-30.3333},{-100,-34},{-114,-34},{-114,60},
                                {160,60}},
                        color={0,0,127},
                        smooth=Smooth.None));
  connect(poleCartConnection.frame_b, poleMassCenter.frame_a) annotation(
      Line(points = {{46, 10}, {78, 10}}, color = {95, 95, 95}, thickness = 0.5, smooth = Smooth.None));
  connect(poleCartConnection.frame_a, rev.frame_b) annotation(
      Line(points = {{26, 10}, {-10, 10}}, color = {95, 95, 95}, thickness = 0.5, smooth = Smooth.None));
  connect(theta, theta) annotation(
      Line(points = {{160, 16}, {160, 16}}, color = {0, 0, 127}, smooth = Smooth.None));
  connect(relativeAngles.angles[3], theta) annotation(
      Line(points = {{-20, -31.6667}, {-20, -34}, {136, -34}, {136, 16}, {160, 16}}, color = {0, 0, 127}, smooth = Smooth.None));
                annotation (
                    experiment(
                        StartTime=1,
                        StopTime=10,
                        __Dymola_Algorithm="Dassl"),
                    Diagram(coordinateSystem(
                            preserveAspectRatio=false,
                            extent={{-150,-100},{150,100}},
                            grid={2,2})),
                    Documentation(info="<html>
    
    Model of a simple double pendulum system. <br>
    The physical Model is used in Modelica_LinearSystems2.Examples.StateSpace.doublePendulumController where it is being
    linearized an used as a base for linear controller design. The results are used to control the crane system
    in Modelica_LinearSystems2.Controller.Examples.DoublePendulum.mo
    
    </html>"),
                    Icon(coordinateSystem(preserveAspectRatio=true, extent={{-150,-100},{150,
                                    100}}), graphics={
                            Rectangle(
                                extent={{-150,122},{150,-120}},
                                lineColor={0,0,0},
                                fillColor={255,255,255},
                                fillPattern=FillPattern.Solid),
                            Rectangle(
                                extent={{-82,22},{82,18}},
                                lineColor={0,0,255},
                                fillPattern=FillPattern.Forward),
                            Rectangle(extent={{-44,54},{0,28}}, lineColor={0,0,0}),
                            Ellipse(
                                extent={{-40,34},{-28,22}},
                                lineColor={0,0,0},
                                fillPattern=FillPattern.Solid,
                                fillColor={255,255,255},
                                lineThickness=0.5),
                            Ellipse(
                                extent={{-16,34},{-4,22}},
                                lineColor={0,0,0},
                                fillColor={255,255,255},
                                fillPattern=FillPattern.Solid,
                                lineThickness=0.5),
                            Line(
                                points={{-18,-16},{10,-62}},
                                color={0,0,0},
                                smooth=Smooth.None),
                            Ellipse(
                                extent={{4,-56},{20,-72}},
                                lineColor={0,0,0},
                                fillColor={0,255,255},
                                fillPattern=FillPattern.Solid),
                            Ellipse(
                                extent={{-25,44},{-19,38}},
                                lineColor={0,0,0},
                                fillColor={95,95,95},
                                fillPattern=FillPattern.Solid),
                            Line(
                                points={{28,46},{4,46}},
                                color={0,0,0},
                                smooth=Smooth.None),
                            Line(
                                points={{34,40},{10,40}},
                                color={0,0,0},
                                smooth=Smooth.None),
                            Line(
                                points={{-22,40},{-18,-16}},
                                color={0,0,0},
                                smooth=Smooth.None),
                            Ellipse(
                                extent={{-20,-15},{-14,-21}},
                                lineColor={0,0,0},
                                fillColor={95,95,95},
                                fillPattern=FillPattern.Solid)}));
            end CartPole;
  
    end ModelicaGym;